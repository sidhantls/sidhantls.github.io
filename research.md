---
layout: page
title: Research
permalink: /research/
---

I'm interested in LLM efficiency, agentic evaluations, evaluation on long-horizon tasks, and self-learning through evaluation traces.

## Publications

<div class="pub" markdown="1">

### Low-Rank Compression of Language Models via Differentiable Rank Selection

<p class="pub-meta" markdown="1">Sidhant Sundrani, **Francesco Tudisco**, **Pasquale Minervini** — University of Edinburgh</p>
<p class="pub-meta" markdown="1">*Proceedings of the Fifteenth Language Resources and Evaluation Conference (LREC 2026), pages 10031–10045. European Language Resources Association (ELRA).*</p>

<p class="pub-links" markdown="1">
[Paper](https://lrec.elra.info/lrec2026-main-787) · [Poster](http://www.lrec-conf.org/proceedings/lrec2026/media/posters/268.pdf) · [Video](http://www.lrec-conf.org/proceedings/lrec2026/media/videos/268.mp4) · [Code](https://github.com/sidhantls/adaptive-rank-selection-svd)
</p>

<details markdown="1">
<summary>Details</summary>

<p align="center">
  <a href="{{site.baseurl}}/images/research/llrc_architecture.png" target="_blank">
    <img class="pub-figure" src="{{site.baseurl}}/images/research/llrc_architecture.png" alt="LLRC training pipeline"/>
  </a>
</p>
*Outline of the training process for learning SVD ranks (click to enlarge).*

Compressing a language model with low-rank decomposition means choosing a rank for every layer — too aggressive and accuracy collapses, too conservative and you save nothing. Existing methods either search a small set of heuristic candidates, or learn ranks with gradient descent but need expensive fine-tuning afterwards to recover performance. We propose **LLRC (Learning to Low-Rank Compress)**: the model is frozen, and only a lightweight differentiable mask over each layer's singular values is trained, so every layer discovers its own compression rate directly — with no fine-tuning after compression.

</details>

<details markdown="1">
<summary>BibTeX</summary>

```
@inproceedings{sundrani2026llrc,
  title     = {Low-Rank Compression of Language Models via Differentiable Rank Selection},
  author    = {Sundrani, Sidhant and Tudisco, Francesco and Minervini, Pasquale},
  booktitle = {Proceedings of the Fifteenth Language Resources and Evaluation Conference (LREC 2026)},
  pages     = {10031--10045},
  year      = {2026},
  publisher = {European Language Resources Association (ELRA)},
  doi       = {10.63317/2xbs948bhby9}
}
```

</details>

</div>

## Ongoing

*Evaluation methods for personalised explanations from language models in education — how to tell whether personalisation genuinely helps a learner.* Work in progress, more soon.
