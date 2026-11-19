"""EDM training of the 'sheet layout -> CT slice' U-Net on the real pairs (bf16, batch 24, AdamW lr 2e-4 with 1000-step warm-up,
EMA 0.9995, P_mean -1.0, P_std 1.4, 30 % of batches from Paris 4, 10 % condition dropout). Automatic resume from points/dernier.pt;
every EVAL_MIN minutes: the 3 judges (judges.py) + a preview sheet. Released weights = EMA at step 6 655 (~40 min on an RTX 5090).
Fine-tune from the released weights: INIT_WEIGHTS=path/to/synthetic_slices_ema_6655.safetensors python entrainer.py
Env: TOTAL_MIN=120 EVAL_MIN=30 PREMIER_MIN=10 BATCH=24 CHURN=30"""
import os, sys, time, json, copy, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from commun import *
import juges
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
dev = 'cuda'
torch.cuda.set_per_process_memory_fraction(16.5 / 32.6)
E = lambda k, d: type(d)(os.environ.get(k, d))
TOTAL, EVAL, PREM, BATCH = (E('TOTAL_MIN', 120.0), E('EVAL_MIN', 30.0), E('PREMIER_MIN', 10.0), E('BATCH', 24))
LR, EMA, PMEAN, PSTD = (0.0002, 0.9995, -1.0, 1.4)
V3_ = os.environ.get('CARTES_V3') == '1'
V2 = os.environ.get('CARTES_V2') == '1' or V3_
SRC = f'{DONNEES}/v3' if V3_ else f'{DONNEES}/v2' if V2 else DONNEES
CHURN = float(os.environ.get('CHURN', 30))
PTS = f'{RACINE}/points'
PL = f'{RACINE}/planches'
os.makedirs(PTS, exist_ok=True)
os.makedirs(PL, exist_ok=True)

class Liberer(Exception):
    pass

def verif():
    if os.path.exists(f'{RACINE}/LIBERER'):
        raise Liberer()

def log(m):
    print(time.strftime('%H:%M:%S'), m, flush=True)

def tri(P):
    q = P['q']
    p4 = P['style'] == STYLES.index('Paris4')
    if V2:
        return (q[:, 2] >= 0.12) & (q[:, 3] <= 0.25) & (q[:, 4] <= 0.25)
    ok_m7 = (q[:, 0] <= 6.5) & (q[:, 1] >= 0.7) & (q[:, 3] <= 0.2) & (q[:, 4] <= 0.2) & (q[:, 5] >= 0.01)
    ok_p4 = (q[:, 3] <= 0.1) & (q[:, 4] <= 0.15) & (q[:, 5] >= 0.01)
    return np.where(p4, ok_p4, ok_m7) & (q[:, 2] >= 0.12)
FICHIERS = set()

def charger():
    import glob
    P = {k: [] for k in ('ct', 'lig', 'dist', 'style', 'axe', 'test', 'bloc', 'q')}
    FICHIERS.clear()
    for f in sorted(glob.glob(f'{SRC}/tuiles_*.npz')):
        FICHIERS.add(f)
        z = np.load(f)
        P['lig'].append(np.unpackbits(z['lig'], axis=2)[:, :, :256])
        for k in ('ct', 'dist', 'style', 'axe', 'test', 'bloc', 'q'):
            P[k].append(z[k])
    P = {k: np.concatenate(v) for k, v in P.items()}
    m = tri(P)
    return ({k: v[m] for k, v in P.items()}, int(len(m)), int(m.sum()))

def main():
    t_dep = time.time()
    P, n_tout, n_garde = charger()
    tr, te = (~P['test'], P['test'])
    log(f'{n_tout} paires, {n_garde} gardées après tri ; entraînement {tr.sum()}, examen {te.sum()}')
    G = {k: torch.tensor(P[k][tr]).pin_memory() for k in ('ct', 'lig', 'dist')}
    M = dict(sty=P['style'][tr].astype(np.int64), axe=P['axe'][tr].astype(np.int64))
    par_style = {}
    POIDS = []

    def classer():
        par_style.clear()
        par_style.update({s: np.nonzero(M['sty'] == s)[0] for s in range(NS) if (M['sty'] == s).any()})
        ks = list(par_style)
        i4 = STYLES.index('Paris4')
        w = np.array([0.3 if k == i4 else 0.7 / max(1, len(ks) - 1) for k in ks]) if i4 in ks and len(ks) > 1 else np.ones(len(ks))
        POIDS[:] = list(w / w.sum())
    classer()
    log("tuiles d'entraînement par style : " + ', '.join((f'{STYLES[s]} {len(v)}' for s, v in par_style.items())))
    rng = np.random.default_rng(5)
    ie = np.nonzero(te)[0]
    sel = []
    for s in range(NS):
        j = ie[P['style'][ie] == s]
        if len(j):
            sel += list(rng.choice(j, min(100 if STYLES[s] == 'Paris4' and V2 else 40, len(j)), replace=False))
    sel = np.array(sorted(sel))
    X = {k: P[k][sel] for k in ('ct', 'lig', 'dist', 'style', 'axe')}
    log(f"""examen : {len(sel)} tuiles ({', '.join((f"{STYLES[s]} {(X['style'] == s).sum()}" for s in range(NS)))})""")
    S = None
    if os.path.exists(f'{DONNEES}/cartes_sim.npz') or os.path.exists(f'{SRC}/cartes_sim.npz'):
        z = np.load(f'{SRC}/cartes_sim.npz' if V3_ and os.path.exists(f'{SRC}/cartes_sim.npz') else f'{DONNEES}/cartes_sim.npz')
        ok = np.load(f'{DONNEES}/v2/cartes_sim_ok.npy') if V2 and (not V3_) and os.path.exists(f'{DONNEES}/v2/cartes_sim_ok.npy') else np.ones(len(z['lig']), bool)
        k_ok = np.nonzero(ok)[0][:128]
        S = dict(lig=np.unpackbits(z['lig'][k_ok], axis=2)[:, :, :256], dist=z['dist'][k_ok])
        m7s = [s for s in range(NS) if STYLES[s] != 'Paris4']
        S['style'] = np.array([m7s[i % len(m7s)] for i in range(len(S['lig']))])
        S['axe'] = np.zeros(len(S['lig']), np.int64)
    fdet = f'{PTS}/detecteur_{os.path.basename(SRC)}.pt' if V2 else f'{PTS}/detecteur.pt'
    p4 = X['style'] == STYLES.index('Paris4')
    if os.path.exists(fdet):
        det = juges.Det().to(dev)
        det.load_state_dict(torch.load(fdet, map_location=dev))
        det.eval()
    else:
        det = juges.entrainer_detecteur(G['ct'], G['lig'], log=log)
        torch.save(det.state_dict(), fdet)
        f_vrai, _ = juges.fidelite(det, X['ct'], X['lig'])
        log(f"détecteur : F1 à 2 px sur les vraies tuiles d'examen = {f_vrai:.3f}")
    G['sig'] = juges.signatures(G['ct'])

    def recharger():
        import glob
        nv = sorted(set(glob.glob(f'{SRC}/tuiles_*.npz')) - FICHIERS)
        n0 = len(M['sty'])
        for f in nv:
            try:
                z = np.load(f)
                m = tri(dict(q=z['q'], style=z['style'])) & ~z['test']
            except Exception:
                continue
            FICHIERS.add(f)
            if not m.any():
                continue
            G['ct'] = torch.cat([G['ct'], torch.tensor(z['ct'][m])]).pin_memory()
            G['dist'] = torch.cat([G['dist'], torch.tensor(z['dist'][m])]).pin_memory()
            G['lig'] = torch.cat([G['lig'], torch.tensor(np.unpackbits(z['lig'][m], axis=2)[:, :, :256])]).pin_memory()
            M['sty'] = np.concatenate([M['sty'], z['style'][m].astype(np.int64)])
            M['axe'] = np.concatenate([M['axe'], z['axe'][m].astype(np.int64)])
            G['sig'] = torch.cat([G['sig'], juges.signatures(torch.tensor(z['ct'][m]))])
        if len(M['sty']) > n0:
            classer()
            log(f"rechargement : +{len(M['sty']) - n0} tuiles d'entraînement -> " + ', '.join((f'{STYLES[s]} {len(v)}' for s, v in par_style.items())))
    net = UNet().to(dev).to(memory_format=torch.channels_last)
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), LR, betas=(0.9, 0.99), weight_decay=0.0)
    st = dict(pas=0, calcul_s=0.0, dernier_eval_s=-1000000000.0, n_eval=0)
    if os.path.exists(f'{PTS}/dernier.pt'):
        c = torch.load(f'{PTS}/dernier.pt', map_location=dev)
        net.load_state_dict(c['net'])
        ema.load_state_dict(c['ema'])
        opt.load_state_dict(c['opt'])
        st = c['st']
        log(f"reprise au pas {st['pas']} ({st['calcul_s'] / 60:.1f} min de calcul)")
    elif os.environ.get('INIT_WEIGHTS'):
        _w = os.environ['INIT_WEIGHTS']
        if _w.endswith('.safetensors'):
            from safetensors.torch import load_file
            _sd = load_file(_w, device='cpu')
        else:
            _sd = torch.load(_w, map_location='cpu', weights_only=True)
        net.load_state_dict(_sd)
        ema.load_state_dict(_sd)
        log(f'fine-tuning from {_w}')
    else:
        log(f'réseau neuf : {sum((p.numel() for p in net.parameters())) / 1000000.0:.1f} M paramètres')
        journal(f"- {time.strftime('%d/%m %H:%M')} UTC : entraînement lancé — {n_tout} paires, {n_garde} gardées après tri qualité, entraînement {int(tr.sum())}, examen {len(sel)} ; réseau {sum((p.numel() for p in net.parameters())) / 1000000.0:.1f} M paramètres, lot {BATCH}, EDM (P_mean {PMEAN}, P_std {PSTD}), bf16.")
    netc = torch.compile(net) if os.environ.get('COMPILE') == '1' else net

    def sauver(chemin=f'{PTS}/dernier.pt'):
        torch.save(dict(net=net.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), st=st), chemin + '.tmp')
        os.replace(chemin + '.tmp', chemin)

    def generer(L, Dd, sty, axe, g, **kw):
        kw.setdefault('churn', CHURN)
        out = []
        for b in range(0, len(L), 48):
            verif()
            cond = condition(torch.tensor(L[b:b + 48], device=dev), torch.tensor(Dd[b:b + 48], device=dev).float() / 4)
            out.append(vers_u8(echantillonner(ema, cond, torch.tensor(sty[b:b + 48], device=dev).long(), torch.tensor(axe[b:b + 48], device=dev).long(), g=g, **kw))[:, 0].cpu().numpy())
        return np.concatenate(out)

    def variantes(R):
        fv = f'{RACINE}/reglages_eval.json'
        if not os.path.exists(fv):
            return
        try:
            V = json.load(open(fv))
        except Exception as e:
            log(f'reglages_eval.json illisible : {e}')
            return
        for v in V.get('variantes', []):
            n = int(v.pop('n', 120))
            nom = v.pop('nom', '_'.join((f'{k}{x}' for k, x in v.items())))
            iv = np.random.default_rng(3).choice(len(sel), min(n, len(sel)), replace=False)
            g = torch.Generator(device=dev).manual_seed(77)
            t1 = time.time()
            gv = generer(X['lig'][iv], X['dist'][iv], X['style'][iv], X['axe'][iv], g, **v)
            R[f'variante_{nom}'] = dict(n=len(iv), realisme=round(juges.juge_realisme(X['ct'][iv], gv)[0] * 100, 1), grain=round(juges.juge_realisme(X['ct'][iv], gv, 'grain')[0] * 100, 1), F1_generees=round(juges.fidelite(det, gv, X['lig'][iv])[0], 3), F1_vraies=round(juges.fidelite(det, X['ct'][iv], X['lig'][iv])[0], 3), minutes=round((time.time() - t1) / 60, 1))
            log(f"variante {nom} : {R[f'variante_{nom}']}")

    def evaluer():
        t0 = time.time()
        ema.eval()
        g = torch.Generator(device=dev).manual_seed(1234)
        gen = []
        for b in range(0, len(sel), 48):
            verif()
            cond = condition(torch.tensor(X['lig'][b:b + 48], device=dev), torch.tensor(X['dist'][b:b + 48], device=dev).float() / 4)
            gen.append(vers_u8(echantillonner(ema, cond, torch.tensor(X['style'][b:b + 48], device=dev).long(), torch.tensor(X['axe'][b:b + 48], device=dev).long(), g=g, churn=CHURN))[:, 0].cpu().numpy())
        gen = np.concatenate(gen)
        gs = None
        if S is not None:
            gs = []
            for b in range(0, len(S['lig']), 48):
                cond = condition(torch.tensor(S['lig'][b:b + 48], device=dev), torch.tensor(S['dist'][b:b + 48], device=dev).float() / 4)
                gs.append(vers_u8(echantillonner(ema, cond, torch.tensor(S['style'][b:b + 48], device=dev).long(), torch.tensor(S['axe'][b:b + 48], device=dev).long(), g=g, churn=CHURN))[:, 0].cpu().numpy())
            gs = np.concatenate(gs)
        t_gen = time.time() - t0
        R = dict(pas=st['pas'], calcul_min=round(st['calcul_s'] / 60, 1))
        for mode in ('brut', 'forme', 'grain'):
            verif()
            a, aa, ft, sc = juges.juge_realisme(X['ct'], gen, mode)
            R[f'realisme_{mode}'] = round(a * 100, 1)
            if mode == 'brut':
                R['realisme_3juges'] = [round(x * 100) for x in aa]
                ft_b, sc_b = (ft, sc)
        if p4.any() and (~p4).sum() > 20:
            R['realisme_m7_seul'] = round(juges.juge_realisme(X['ct'][~p4], gen[~p4])[0] * 100, 1)
        if gs is not None:
            R['realisme_cartes_sim'] = round(juges.juge_realisme(X['ct'][~p4], gs)[0] * 100, 1)
        fv, fvs = juges.fidelite(det, X['ct'], X['lig'])
        fg, fgs = juges.fidelite(det, gen, X['lig'])
        R.update(F1_vraies=round(fv, 3), F1_generees=round(fg, 3), F1_vraies_m7=round(float(fvs[~p4].mean()), 3), F1_generees_m7=round(float(fgs[~p4].mean()), 3), F1_vraies_paris4=round(float(fvs[p4].mean()), 3), F1_generees_paris4=round(float(fgs[p4].mean()), 3), F1bande_vraies=round(juges.fidelite(det, X['ct'], X['lig'], bande=True)[0], 3), F1bande_generees=round(juges.fidelite(det, gen, X['lig'], bande=True)[0], 3))
        if gs is not None:
            R['F1bande_cartes_sim'] = round(juges.fidelite(det, gs, S['lig'], bande=True)[0], 3)
        if gs is not None:
            R['F1_cartes_sim'] = round(juges.fidelite(det, gs, S['lig'])[0], 3)
        cg, cv = (juges.plus_proche(gen, G['sig']), juges.plus_proche(X['ct'], G['sig']))
        R.update(copie_generees_med=round(float(np.median(cg)), 3), copie_generees_max=round(float(cg.max()), 3), copie_vraies_med=round(float(np.median(cv)), 3), copie_vraies_max=round(float(cv.max()), 3))
        if gs is not None:
            cs = juges.plus_proche(gs, G['sig'])
            R['copie_sim_max'] = round(float(cs.max()), 3)
        R['cartes_examen'] = juges.continuite_cartes(X['lig'])
        if gs is not None:
            R['cartes_sim'] = juges.continuite_cartes(S['lig'])
        R['trace_vraies'] = juges.details(det, X['ct'], X['lig'])
        R['trace_generees'] = juges.details(det, gen, X['lig'])
        if gs is not None:
            R['trace_sim'] = juges.details(det, gs, S['lig'])
        variantes(R)
        R['minutes_eval'] = round((time.time() - t0) / 60, 1)
        R['minutes_generation'] = round(t_gen / 60, 1)
        planche(gen, gs, ft_b, sc_b)
        np.savez_compressed(f"{PL}/gen_p{st['pas']:06d}.npz", gen=gen, sel=sel, gs=gs if gs is not None else np.zeros(0))
        log('JUGES ' + json.dumps(R, ensure_ascii=False))
        with open(f'{RACINE}/juges.jsonl', 'a') as f:
            f.write(json.dumps(R) + '\n')
        journal(f"- {time.strftime('%d/%m %H:%M')} UTC · POINT pas {st['pas']} ({R['calcul_min']} min de calcul) : réalisme AUC {R['realisme_brut']} % (forme {R['realisme_forme']} %, grain {R['realisme_grain']} %, cartes du simulateur {R.get('realisme_cartes_sim', '-')} %) · tracé F1 vraies {R['F1_vraies']} / générées {R['F1_generees']} / simulateur {R.get('F1_cartes_sim', '-')} · copie : cosinus max généré {R['copie_generees_max']} (médiane {R['copie_generees_med']}) contre vraies d'examen {R['copie_vraies_max']} ({R['copie_vraies_med']})" + f" · cartes{(' V2' if V2 else '')} : continuité examen {R['cartes_examen']['cont']}, simulateur {R.get('cartes_sim', {}).get('cont', '-')} · feuilles retrouvées (vraies / générées / simulateur) : continuité {R['trace_vraies']['cont_det']} / {R['trace_generees']['cont_det']} / {R.get('trace_sim', {}).get('cont_det', '-')}, trait sans feuille {R['trace_vraies']['trait_sans_feuille']} / {R['trace_generees']['trait_sans_feuille']} / {R.get('trace_sim', {}).get('trait_sans_feuille', '-')}, feuille sans trait {R['trace_vraies']['feuille_sans_trait']} / {R['trace_generees']['feuille_sans_trait']} / {R.get('trace_sim', {}).get('feuille_sans_trait', '-')}")
        return R
    _evaluer = evaluer

    def evaluer():
        try:
            return _evaluer()
        except Liberer:
            sauver()
            log('LIBERER pendant un point des juges : point sauvé, je me ferme')
            sys.exit(3)

    def planche(gen, gs, ft, sc):
        import cv2
        r2 = np.random.default_rng(st['pas'])
        k = r2.choice(len(sel), 8, replace=False)

        def ov(t, l):
            g = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
            g[l > 0] = (66, 228, 240)
            return g
        rows = [np.hstack([cv2.cvtColor(X['ct'][i], cv2.COLOR_GRAY2BGR) for i in k]), np.hstack([cv2.cvtColor(gen[i], cv2.COLOR_GRAY2BGR) for i in k]), np.hstack([ov(gen[i], X['lig'][i]) for i in k])]
        if gs is not None:
            rows += [np.hstack([cv2.cvtColor(gs[i], cv2.COLOR_GRAY2BGR) for i in range(8)]), np.hstack([ov(gs[i], S['lig'][i]) for i in range(8)])]
        pl = np.vstack(rows)
        cv2.imwrite(f"{PL}/planche_p{st['pas']:06d}.jpg", pl, [cv2.IMWRITE_JPEG_QUALITY, 90])
        gr = cv2.cvtColor(pl, cv2.COLOR_BGR2GRAY)
        ex = [gr[0:256, :512], gr[256:512, :512]] + ([gr[768:1024, :512]] if gs is not None else [])
        cv2.imwrite(f"{PL}/voir_p{st['pas']:06d}.jpg", np.vstack(ex), [cv2.IMWRITE_JPEG_QUALITY, 75])
    rng = np.random.default_rng()
    t_last = time.time()
    durees = []
    while True:
        if os.path.exists(f'{RACINE}/LIBERER'):
            sauver()
            log('LIBERER : point sauvé, je me ferme')
            sys.exit(3)
        if os.path.exists(f'{RACINE}/STOP') or st['calcul_s'] >= TOTAL * 60:
            sauver()
            evaluer()
            sauver()
            torch.save(ema.state_dict(), f"{PTS}/ema_p{st['pas']:06d}.pt")
            log('FIN')
            return
        if os.path.exists(f'{RACINE}/EVALUER'):
            os.remove(f'{RACINE}/EVALUER')
            sauver()
            evaluer()
            torch.save(ema.state_dict(), f"{PTS}/ema_p{st['pas']:06d}.pt")
            t_last = time.time()
        if st['n_eval'] == 0 and st['calcul_s'] >= PREM * 60 or (st['calcul_s'] - st['dernier_eval_s'] >= EVAL * 60 and st['n_eval'] > 0):
            sauver()
            evaluer()
            st['dernier_eval_s'] = st['calcul_s']
            st['n_eval'] += 1
            sauver()
            torch.save(ema.state_dict(), f"{PTS}/ema_p{st['pas']:06d}.pt")
            t_last = time.time()
        if st['pas'] % 1000 == 0:
            recharger()
        ss = rng.choice(list(par_style), BATCH, p=POIDS)
        idx = np.array([rng.choice(par_style[s]) for s in ss])
        it = torch.tensor(idx)
        ct, lig, dist = (G[k][it].to(dev, non_blocking=True) for k in ('ct', 'lig', 'dist'))
        dist = dist.float() / 4
        sty = torch.tensor(M['sty'][idx], device=dev)
        axe = torch.tensor(M['axe'][idx], device=dev)
        xs = torch.stack([ct.float(), lig.float(), dist], 1)
        out = []
        for i in range(BATCH):
            v = xs[i]
            if rng.random() < 0.5:
                v = v.flip(2)
            if rng.random() < 0.5:
                v = v.flip(1)
            if M['axe'][idx[i]] == 0:
                v = torch.rot90(v, int(rng.integers(4)), (1, 2))
            out.append(v)
        xs = torch.stack(out)
        x0 = (xs[:, :1] / 127.5 - 1).contiguous(memory_format=torch.channels_last)
        cond = condition(xs[:, 1], xs[:, 2])
        dc = torch.rand(BATCH, device=dev) < 0.1
        cond = torch.where(dc[:, None, None, None], torch.zeros_like(cond), cond)
        ds = torch.rand(BATCH, device=dev) < 0.1
        sty = torch.where(ds, torch.full_like(sty, NS), sty)
        axe = torch.where(ds, torch.full_like(axe, 2), axe)
        sig = torch.exp(torch.randn(BATCH, device=dev) * PSTD + PMEAN)
        D = debruiter(netc, x0 + torch.randn_like(x0) * sig.view(-1, 1, 1, 1), sig, cond, sty, axe)
        w = (sig ** 2 + SD ** 2) / (sig * SD) ** 2
        loss = (w.view(-1, 1, 1, 1) * (D - x0) ** 2).mean()
        for gp in opt.param_groups:
            gp['lr'] = LR * min(1.0, (st['pas'] + 1) / 1000)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            d = min(EMA, (1 + st['pas']) / (10 + st['pas']))
            for pe, pn in zip(ema.parameters(), net.parameters()):
                pe.lerp_(pn, 1 - d)
        st['pas'] += 1
        now = time.time()
        dt = now - t_last
        t_last = now
        med = float(np.median(durees[-200:])) if len(durees) > 20 else dt
        if dt > 5 * med + 1:
            dt = med
        durees.append(dt)
        st['calcul_s'] += dt
        if st['pas'] % 200 == 0:
            log(f"pas {st['pas']} perte {loss.item():.4f} norme {gn.item():.2f} · {1 / med:.1f} pas/s · calcul {st['calcul_s'] / 60:.1f} min · mémoire carte {torch.cuda.max_memory_reserved() / 1000000000.0:.1f} Go")
        if st['pas'] % 2000 == 0:
            sauver()
if __name__ == '__main__':
    main()
