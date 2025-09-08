# A Bidirectional Selective State Space Model for Imaging-Genetic Causal Analysis of Neurodegenerative Diseases

Source codes for the paper "A Bidirectional Selective State Space Model for Imaging-Genetic Causal Analysis of Neurodegenerative Diseases".

## Task Overview

![Task](overall.png | width=500)

## Training

```bash
# The training process of MRI and SNP feature extraction and causal inference
python train_clip.py
# The training process of classification
python train_cls.py
```
## Testing

```bash
# The testing process of SNP classification
python test_cls.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
