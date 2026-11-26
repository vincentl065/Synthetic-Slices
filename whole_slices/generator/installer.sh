#!/bin/bash
# sur la machine louée : dépendances (torch est supposé présent dans l'image) puis contrôle
pip install -q --break-system-packages numpy scipy opencv-python-headless scikit-image psutil 2>&1 | grep -v WARNING | tail -2
python -c "import torch, cv2, scipy, skimage; print('torch', torch.__version__, '| cartes', torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
nproc; free -g | sed -n 2p; df -h . | tail -1
