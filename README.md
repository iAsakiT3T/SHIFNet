## Unveiling the Potential of Segment Anything Model 2 for RGB-Thermal Semantic Segmentation with Language Guidance

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unveiling-the-potential-of-segment-anything/thermal-image-segmentation-on-pst900)](https://paperswithcode.com/sota/thermal-image-segmentation-on-pst900?p=unveiling-the-potential-of-segment-anything)

![framework](assets/framework.png)

SHIFNet is an innovative SAM2-driven Hybrid Interactive Fusion Paradigm designed for RGB-T perception tasks. This framework fully unlocks the potential of SAM2 through language-guided adaptation, effectively mitigating its inherent RGB bias and enhancing cross-modal semantic consistency. SHIFNet consists of two key components: (1) Semantic-Aware Cross-modal Fusion (SACF) module, which dynamically balances modality contributions through text-guided affinity learning, enabling adaptive cross-modal information integration; (2) Heterogeneous Prompting Decoder (HPD), which enhances global semantic understanding through a semantic enhancement module and category embeddings, ensuring cross-modal semantic consistency. With only 32.27M trainable parameters, SHIFNet achieves 89.8%, 67.8%, and 59.2% mIoU on PST900, FMB, and MFNet benchmarks, respectively, while attaining 76.5% pedestrian detection accuracy in safety-critical scenarios. By reducing the cost of large-scale data collection and enhancing multi-modal perception capabilities, SHIFNet provides a reliable perception foundation for intelligent robotic systems operating in complex environments.

## Visualization on FMB dataset
![vis](assets/vis.png)

## 🤝 Publication:
Please consider referencing this paper if you use the ```code``` from our work.
Thanks a lot :)

```
@article{zhao2025unveiling,
  title={Unveiling the Potential of Segment Anything Model 2 for RGB-Thermal Semantic Segmentation with Language Guidance},
  author={Zhao, Jiayi and Teng, Fei and Luo, Kai and Zhao, Guoqiang and Li, Zhiyong and Zheng, Xu and Yang, Kailun},
  journal={arXiv preprint arXiv:2503.02581},
  year={2025}
}
```
