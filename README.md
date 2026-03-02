# DeepAFL
DeepAFL: Deep Analytic Federated Learning. Accepted in the Fourteenth International Conference on Learning Representations (ICLR 2026).

## ⚙️ Environment Setup
We recommend that `python=3.12`, `pytorch=2.5.1`, and `CUDA=12.4` are installed in your environment. To install the required packages, run the following command:

```
pip install -r requirements.txt
```

## 🚀 Usage
To begin training, run the following command:

```
bash run.sh
```

## 📌 Notes

Due to hardware-dependent randomness and numerical precision, you may observe slight numerical deviations in the reported results. In rare cases, such deviations can affect the optimization procedure and lead to numerical instability, e.g., encountering singular matrices during solving.

If you meet such issues, you can try the following workarounds:

* **Increase the regularization coefficients `REG` or `REG_SAND`**, which may help improve numerical stability and reduce the chance of singularity.

* **Alternatively, rerun with a different random seed**, which may help bypass the problematic case.

## ✒️ Citation

If you find our work useful for your research, please consider citing our paper:

```
@inproceedings{DeepAFL,
  title={{DeepAFL}: Deep Analytic Federated Learning},
  author={Jianheng Tang and Yajiang Huang and Kejia Fan and Feijiang Han and Jiaxu Li and Jinfeng Xu and Run He and Anfeng Liu and Houbing Herbert Song and Huiping Zhuang and Yunhuai Liu},
  booktitle={The Fourteenth International Conference on Learning Representations (ICLR)},
  year={2026},
  url={https://openreview.net/forum?id=ve3EzAvMGe}
}
```

## ✉️ Contact 
If you have any questions or suggestions regarding our work, please feel free to contact us via email at `tangentheng@gmail.com`.
