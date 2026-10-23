"""Builds paper/mini_jev_paper.docx. Needs python-docx (pip install python-docx).

    python paper/build_paper.py

Tables of numbers come from results/study/*.md; figures from paper/figures (python paper/make_figures.py).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from docx.enum.text import WD_ALIGN_PARAGRAPH  # noqa: E402

from docx_helpers import Paper  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(HERE, "figures")
C = WD_ALIGN_PARAGRAPH.CENTER

P = Paper("What does fine-tuning buy a small vision-language decision model?")
P.refs = {
    "bai": "Bai, S., et al. Qwen3-VL Technical Report. arXiv:2511.21631, 2025.",
    "hu": "Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. LoRA: Low-rank adaptation of large language models. ICLR 2022 (arXiv:2106.09685).",
    "guo": "Guo, C., Pleiss, G., Sun, Y., and Weinberger, K. Q. On calibration of modern neural networks. ICML 2017.",
    "naeini": "Naeini, M. P., Cooper, G. F., and Hauskrecht, M. Obtaining well calibrated probabilities using Bayesian binning. AAAI 2015.",
    "epstein": "Epstein, E. S. A scoring system for probability forecasts of ranked categories. Journal of Applied Meteorology 8(6):985-987, 1969.",
    "efron": "Efron, B. Bootstrap methods: another look at the jackknife. The Annals of Statistics 7(1):1-26, 1979.",
    "kadavath": "Kadavath, S., Conerly, T., Askell, A., Henighan, T., et al. Language models (mostly) know what they know. arXiv:2207.05221, 2022.",
    "zheng": "Zheng, C., Zhou, H., Meng, F., Zhou, J., and Huang, M. Large language models are not robust multiple choice selectors. ICLR 2024.",
    "robinson": "Robinson, J., Rytting, C. M., and Wingate, D. Leveraging large language models for multiple choice question answering. ICLR 2023 (arXiv:2210.12353).",
    "pezeshkpour": "Pezeshkpour, P., and Hruschka, E. Large language models sensitivity to the order of options in multiple-choice questions. arXiv:2308.11483, 2023.",
    "salsa": "Berdichevsky, R., Nahum-Gefen, S., and Ben Zaken, E. SALSA: Single-pass autoregressive LLM structured classification. arXiv:2510.22691, 2025.",
    "sood": "Sood, B., and Savelka, J. Punching above their weight: Classification-head fine-tuning of tiny language models (TLMs) for verifiable multiple-choice tasks. arXiv:2607.03801, 2026.",
    "yousefi": "Yousefiramandi, A., and Cooney, C. Fine-tuning causal LLMs for text classification: Embedding-based vs. instruction-based approaches. arXiv:2512.12677, 2025.",
    "wei": "Wei, J., Bosma, M., Zhao, V. Y., Guu, K., Yu, A. W., Lester, B., Du, N., Dai, A. M., and Le, Q. V. Finetuned language models are zero-shot learners. ICLR 2022 (arXiv:2109.01652).",
    "sanh": "Sanh, V., Webson, A., Raffel, C., Bach, S. H., et al. Multitask prompted training enables zero-shot task generalization. ICLR 2022.",
    "chung": "Chung, H. W., Hou, L., Longpre, S., Zoph, B., et al. Scaling instruction-finetuned language models. arXiv:2210.11416, 2022.",
    "jevblog": "Jev, TypeSafe AI and System One: what is publicly known. OrcaRouter blog. https://www.orcarouter.ai/blog/jev-typesafe-system-one-what-we-know (accessed October 2026).",
    "diOverview": "Summary of open Jev-like decision models compared on the Decision Index and JevBench. PyTorch Korea forum. https://discuss.pytorch.kr/t/jev-decision-index-jevbench/12095 (accessed October 2026).",
    "diKit": "Decision Index reproduction kit. https://github.com/apolinario/decision-index (accessed October 2026).",
    "diNews": "Jev Decision Index ranks 30+ open-weight decision models. AI Weekly. https://aiweekly.co/alerts/jev-decision-index-ranks-30-open-weight-decision-models (accessed October 2026).",
    "decider": "Mapika. decider-2b: typed decisions with calibrated probabilities in one forward pass (model card). https://huggingface.co/Mapika/decider-2b (accessed October 2026).",
    "typic": "typic-bert: an encoder-based decision model (preprint). Research Square, rs-11162889. https://www.researchsquare.com/article/rs-11162889 (accessed October 2026).",
    # ---- dataset sources (appendix) ----
    "bowman": "Bowman, S. R., Angeli, G., Potts, C., and Manning, C. D. A large annotated corpus for learning natural language inference. EMNLP 2015.",
    "williams": "Williams, A., Nangia, N., and Bowman, S. R. A broad-coverage challenge corpus for sentence understanding through inference. NAACL-HLT 2018.",
    "glue": "Wang, A., Singh, A., Michael, J., Hill, F., Levy, O., and Bowman, S. R. GLUE: A multi-task benchmark and analysis platform for natural language understanding. ICLR 2019.",
    "superglue": "Wang, A., Pruksachatkun, Y., Nangia, N., Singh, A., Michael, J., Hill, F., Levy, O., and Bowman, S. R. SuperGLUE: A stickier benchmark for general-purpose language understanding systems. NeurIPS 2019.",
    "arc": "Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A., Schoenick, C., and Tafjord, O. Think you have solved question answering? Try ARC, the AI2 reasoning challenge. arXiv:1803.05457, 2018.",
    "tweeteval": "Barbieri, F., Camacho-Collados, J., Espinosa Anke, L., and Neves, L. TweetEval: Unified benchmark and comparative evaluation for tweet classification. Findings of EMNLP 2020.",
    "scienceqa": "Lu, P., Mishra, S., Xia, T., Qiu, L., Chang, K.-W., Zhu, S.-C., Tafjord, O., Clark, P., and Kalyan, A. Learn to explain: Multimodal reasoning via thought chains for science question answering. NeurIPS 2022.",
    "aokvqa": "Schwenk, D., Khandelwal, A., Clark, C., Marino, K., and Mottaghi, R. A-OKVQA: A benchmark for visual question answering using world knowledge. ECCV 2022.",
    "chartqa": "Masry, A., Long, D. X., Tan, J. Q., Joty, S., and Hoque, E. ChartQA: A benchmark for question answering about charts with visual and logical reasoning. Findings of ACL 2022.",
    "food101": "Bossard, L., Guillaumin, M., and Van Gool, L. Food-101: Mining discriminative components with random forests. ECCV 2014.",
    "openbookqa": "Mihaylov, T., Clark, P., Khot, T., and Sabharwal, A. Can a suit of armor conduct electricity? A new dataset for open book question answering. EMNLP 2018.",
    "sciq": "Welbl, J., Liu, N. F., and Gardner, M. Crowdsourcing multiple choice science questions. Workshop on Noisy User-generated Text, 2017.",
    "race": "Lai, G., Xie, Q., Liu, H., Yang, Y., and Hovy, E. RACE: Large-scale reading comprehension dataset from examinations. EMNLP 2017.",
    "cifar": "Krizhevsky, A. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 2009.",
    "mnist": "LeCun, Y., Bottou, L., Bengio, Y., and Haffner, P. Gradient-based learning applied to document recognition. Proceedings of the IEEE 86(11):2278-2324, 1998.",
    "fashion": "Xiao, H., Rasul, K., and Vollgraf, R. Fashion-MNIST: a novel image dataset for benchmarking machine learning algorithms. arXiv:1708.07747, 2017.",
    "banking": "Casanueva, I., Temcinas, T., Gerz, D., Henderson, M., and Vulic, I. Efficient intent detection with dual sentence encoders. Workshop on NLP for Conversational AI, 2020.",
    "goemotions": "Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., and Ravi, S. GoEmotions: A dataset of fine-grained emotions. ACL 2020.",
    "imdb": "Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., and Potts, C. Learning word vectors for sentiment analysis. ACL 2011.",
    "rotten": "Pang, B., and Lee, L. Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales. ACL 2005.",
    "sst": "Socher, R., Perelygin, A., Wu, J., Chuang, J., Manning, C. D., Ng, A., and Potts, C. Recursive deep models for semantic compositionality over a sentiment treebank. EMNLP 2013.",
    "zhang15": "Zhang, X., Zhao, J., and LeCun, Y. Character-level convolutional networks for text classification. NeurIPS 2015.",
    "csqa": "Talmor, A., Herzig, J., Lourie, N., and Berant, J. CommonsenseQA: A question answering challenge targeting commonsense knowledge. NAACL-HLT 2019.",
    "boolq": "Clark, C., Lee, K., Chang, M.-W., Kwiatkowski, T., Collins, M., and Toutanova, K. BoolQ: Exploring the surprising difficulty of natural yes/no questions. NAACL-HLT 2019.",
    "qasc": "Khot, T., Clark, P., Guerquin, M., Jansen, P., and Sabharwal, A. QASC: A dataset for question answering via sentence composition. AAAI 2020.",
    "quartz": "Tafjord, O., Gardner, M., Lin, K., and Clark, P. QuaRTz: An open-domain dataset of qualitative relationship questions. EMNLP 2019.",
    "cola": "Warstadt, A., Singh, A., and Bowman, S. R. Neural network acceptability judgments. Transactions of the ACL 7:625-641, 2019.",
    "stsb": "Cer, D., Diab, M., Agirre, E., Lopez-Gazpio, I., and Specia, L. SemEval-2017 task 1: Semantic textual similarity multilingual and cross-lingual focused evaluation. SemEval 2017.",
    "sms": "Almeida, T. A., Gomez Hidalgo, J. M., and Yamakami, A. Contributions to the study of SMS spam filtering: new collection and results. ACM Symposium on Document Engineering, 2011.",
    "pets": "Parkhi, O. M., Vedaldi, A., Zisserman, A., and Jawahar, C. V. Cats and dogs. CVPR 2012.",
    "flowers": "Nilsback, M.-E., and Zisserman, A. Automated flower classification over a large number of classes. ICVGIP 2008.",
    "eurosat": "Helber, P., Bischke, B., Dengel, A., and Borth, D. EuroSAT: A novel dataset and deep learning benchmark for land use and land cover classification. IEEE J-STARS 12(7):2217-2226, 2019.",
    "dtd": "Cimpoi, M., Maji, S., Kokkinos, I., Mohamed, S., and Vedaldi, A. Describing textures in the wild. CVPR 2014.",
    "gtsrb": "Stallkamp, J., Schlipsing, M., Salmen, J., and Igel, C. Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition. Neural Networks 32:323-332, 2012.",
    "resisc": "Cheng, G., Han, J., and Lu, X. Remote sensing image scene classification: Benchmark and state of the art. Proceedings of the IEEE 105(10):1865-1883, 2017.",
    "stl10": "Coates, A., Ng, A., and Lee, H. An analysis of single-layer networks in unsupervised feature learning. AISTATS 2011.",
    "svhn": "Netzer, Y., Wang, T., Coates, A., Bissacco, A., Wu, B., and Ng, A. Y. Reading digits in natural images with unsupervised feature learning. NIPS Workshop on Deep Learning and Unsupervised Feature Learning, 2011.",
    "mrpc": "Dolan, W. B., and Brockett, C. Automatically constructing a corpus of sentential paraphrases. Workshop on Paraphrasing, 2005.",
    "rte": "Dagan, I., Glickman, O., and Magnini, B. The PASCAL recognising textual entailment challenge. Machine Learning Challenges Workshop, 2005.",
    "cb": "de Marneffe, M.-C., Simons, M., and Tonhauser, J. The CommitmentBank: Investigating projection in naturally occurring discourse. Sinn und Bedeutung 23, 2019.",
    "copa": "Roemmele, M., Bejan, C. A., and Gordon, A. S. Choice of plausible alternatives: An evaluation of commonsense causal reasoning. AAAI Spring Symposium, 2011.",
    "wic": "Pilehvar, M. T., and Camacho-Collados, J. WiC: The word-in-context dataset for evaluating context-sensitive meaning representations. NAACL-HLT 2019.",
    "squad": "Rajpurkar, P., Zhang, J., Lopyrev, K., and Liang, P. SQuAD: 100,000+ questions for machine comprehension of text. EMNLP 2016.",
    "carer": "Saravia, E., Liu, H.-C. T., Huang, Y.-H., Wu, J., and Chen, Y.-S. CARER: Contextualized affect representations for emotion recognition. EMNLP 2018.",
    "lang": "Lang, K. NewsWeeder: Learning to filter netnews. ICML 1995.",
}


def md_table(path):
    """Header and rows of the first markdown table in a file."""
    rows = []
    for line in open(path, encoding="utf-8"):
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue
            rows.append(cells)
        elif rows:
            break
    return rows[0], rows[1:]


# ====================================================================== title block
t = P.p("**What Does Fine-Tuning Buy a Small Vision-Language Decision Model?**", align=C, size=17, after=2)
P.p("**A controlled study against calibrated baselines**", align=C, size=13, after=8)
P.p("Astitwa Agarwal", align=C, size=12, after=0)
P.p("Independent researcher", align=C, size=10.5, after=0)
P.p("Code: github.com/agarwalastitwa027-svg/mini-jev-multimodal", align=C, size=9, after=0)
P.p("Model and data: huggingface.co/AstitwaAgarwal/multimodal-jev", align=C, size=9, after=10)

P.p("**Abstract.** A decision model answers typed questions about a state made of text and images by returning a "
    "probability for every allowed option in one forward pass, without generating text. We study what fine-tuning a small "
    "vision-language model (Qwen3-VL-2B with LoRA and a learned option-scoring head) buys over the baselines a practitioner "
    "would try first, using an open implementation that trains on an 8 GB laptop GPU. Measured against the base model with "
    "post-hoc temperature scaling, fine-tuning raises accuracy by 6.9 points (95% CI 5.5 to 8.3) on 16 held-out datasets, but "
    "by only 2.8 points (1.2 to 4.4) on 11 further datasets that influenced no design decision, and not significantly on three "
    "zero-shot sources. Temperature scaling alone removes 74 to 87% of the calibration error that separates the raw base model "
    "from the fine-tuned one, so most of the calibration benefit that is easy to attribute to fine-tuning is not due to it. "
    "Over three seeds, training on a broader mix of tasks beats a narrower mix on all five held-out collections (+1.3 to +5.1 "
    "points), whereas a learned answer head is no more accurate than a plain letter-logit readout, which generalises better to "
    "unseen sources but cannot handle more than 26 options. The fine-tuned model falls below the base model on questions with "
    "more than ten options; averaging over option orders recovers part of the gap. We release the code, the model, five test "
    "collections and every per-question prediction.", size=10, indent=0.4, after=4)
P.p("**Keywords:** decision models; calibration; vision-language models; LoRA; multiple-choice questions; temperature scaling",
    size=10, indent=0.4, after=6)

# ====================================================================== 1 Introduction
P.h("1  Introduction")
P.p("Many practical systems need a model that chooses among a fixed set of options and says how sure it is: routing a support "
    "ticket, deciding whether a message needs review, selecting a tool, grading an answer against a rubric. A generative "
    "language model can be asked to do this and its text parsed, but the answer is slow to produce and carries no usable "
    "confidence. A ~~decision model~~ instead reads the whole request once and returns a probability for each allowed option, "
    "which makes a single forward pass sufficient and lets downstream code threshold, route or abstain. The idea was popularised "
    "by TypeSafe AI's Jev, a closed model whose architecture and weights are unpublished [@jevblog], and by the Decision "
    "Index, a public benchmark and leaderboard for typed decisions with an open reproduction kit [@diKit]. Several open "
    "models in the same style have since appeared, for example decider-2b [@decider] and an encoder-based model [@typic], but "
    "as far as we could find none is accompanied by a controlled study that separates what the training contributes from what a "
    "careful baseline already provides.")
P.p("This paper asks what fine-tuning a small vision-language model into a decision model actually buys. The baselines matter. "
    "The simplest one reads the next-token logits of the option letters from the untouched base model. A stronger and equally "
    "cheap one calibrates those probabilities afterwards with temperature scaling [@guo]. Comparisons that omit the second "
    "baseline compare a miscalibrated model with a calibrated one and credit the difference to training.")
P.p("We organise the study around five research questions. **RQ1:** how much accuracy does fine-tuning add over the base model "
    "once the base model is calibrated, and does the gain hold on datasets that played no role in any design choice? **RQ2:** "
    "how much of the calibration improvement is attributable to fine-tuning rather than to temperature scaling? **RQ3:** does "
    "training on more tasks improve generalisation, measured over several seeds? **RQ4:** does a learned option-scoring head "
    "beat reading letter logits when everything else is held fixed? **RQ5:** where does the fine-tuned model fail, and do "
    "cheap inference-time or data-side remedies help?")
P.p("The answers are mixed, and we report them as such. Fine-tuning helps, but by less than half as much on genuinely new datasets as on "
    "the datasets used while developing the method; most of the calibration gain is available for free; breadth of training "
    "tasks is the one change that helped consistently; the learned head is justified by calibration and by supporting many options, "
    "not by accuracy; and the fine-tuned model is weaker than the base model on questions with many options.")
P.p("**Contributions.** (i) A controlled comparison of a fine-tuned small vision-language decision model against a "
    "temperature-scaled base model and a letter-readout head, with paired bootstrap intervals and three seeds per recipe. "
    "(ii) A confirmation collection of 13 datasets assembled after the comparison, so that the final claim rests on data that "
    "influenced no decision. (iii) An analysis by number of options that exposes a weakness, and measurements of two "
    "remedies, one of which fails. (iv) An open, laptop-scale implementation with evaluation tools and all per-question "
    "predictions. We make no claim of architectural novelty; the contribution is empirical.")

# ====================================================================== 2 Related work
P.h("2  Related work")
P.p("**Decision models and their benchmarks.** The Decision Index defines typed questions (choose one option, estimate the "
    "probability of a yes/no proposition) over a frozen suite of public benchmarks and scores models with chance-corrected, "
    "coverage-adjusted metrics [@diKit,@diNews]. Entries range from fine-tuned multi-billion-parameter models to training-free "
    "readouts of option-token probabilities [@diOverview]. These are model cards and repositories rather than papers, and their evaluations "
    "are reported by their authors; our study complements them by controlling the baselines and reporting uncertainty.")
P.p("**Classification heads and letter readouts.** Presenting all options and reading the label the model prefers works well for "
    "models that can bind a symbol to its option [@robinson]. SALSA maps each class to a distinct output token and reads only "
    "those logits in one pass, with LoRA fine-tuning [@salsa]. Adding a discriminative head to a causal language model's final "
    "representation has been compared with label generation for multiple-choice tasks on small Qwen3 models [@sood] and for "
    "document classification [@yousefi]. Our letter-readout baseline is the SALSA-style readout placed in the same framework as our "
    "answer head (same data, prompts, caching, calibration), which lets us isolate the head; unlike prior comparisons we evaluate on "
    "datasets unseen in training and report intervals.")
P.p("**Calibration.** Modern networks are often overconfident, and a single temperature fitted on held-out data repairs "
    "much of this [@guo]; expected calibration error (ECE) summarises the remaining gap [@naeini]. Large language models can "
    "be well calibrated on multiple-choice questions when the format suits them [@kadavath]. We measure rather than assume: "
    "for the small base model in our format, raw confidence is poor and a temperature recovers most of the gap.")
P.p("**Option-order sensitivity.** Models favour particular option labels and positions, so reordering options changes answers "
    "[@zheng,@pezeshkpour]. PriDe estimates and removes the prior over option IDs at inference time [@zheng]. The inference-time "
    "technique we test, averaging probabilities over several option orders, is the simplest member of this family.")
P.p("**Task breadth.** Fine-tuning on many tasks written as instructions improves zero-shot performance on held-out tasks "
    "[@wei,@sanh], and the benefit grows with the number of tasks [@chung]. Our broad-versus-narrow comparison tests whether "
    "this holds at a much smaller scale (12 versus 24 training sources, one 2B model). The backbone is Qwen3-VL [@bai] and the "
    "adaptation method is LoRA [@hu].")

# ====================================================================== 3 Method
P.h("3  Method")
P.h("3.1  Problem format", 2)
P.p("A request consists of a ~~state~~ (text or JSON, plus up to eight images) and a set of typed questions about it. A "
    "~~choice~~ question has between 2 and 255 options, each a short key with an optional description. A ~~noul~~ question is a "
    "yes/no proposition and returns P(yes). A ~~score~~ question has ordered levels and returns a probability for each level and "
    "the expected level. For every question the model returns a probability for each allowed option; it cannot answer outside "
    "them. Requests longer than a declared limit (16,384 tokens), with more than 255 options or more than eight images are "
    "refused rather than truncated.")
P.h("3.2  Model", 2)
P.p("The backbone is Qwen3-VL-2B-Instruct [@bai] with its language-model head removed, so it outputs hidden states. LoRA [@hu] "
    "(rank 16, alpha 32, dropout 0.05) adapts all linear projections of the text decoder; the vision tower is frozen. The "
    "images and state form a shared prefix that is run once; each question is a short suffix (the question text and the "
    "options) appended to the cached prefix, and all questions about a state are answered in one batched pass. In training and in "
    "inference the prefix and suffix are tokenised separately and concatenated identically, so the cached path equals the "
    "full-sequence path (checked to within 1e-5 in float32 in the test suite).")
P.p("The answer head scores each option from its own text. Let h_last be the final hidden state at the last suffix token and "
    "h_j the mean hidden state over the tokens of option j. With two small MLPs (a LayerNorm, a hidden layer and a 256-dimensional "
    "projection),")
P.equation("q = W_q( h_last ),     o_j = W_o( mean of h over option j )", 1)
P.equation("s_j = ⟨ q, o_j ⟩ / √d,     p_j = softmax_j ( s_j / T_τ )", 2)
P.p("where τ is the question type (choice, noul or score) and T_τ is a temperature fitted per type by minimising the "
    "negative log-likelihood on a validation set after training. Because options are scored through their own text, new "
    "options need no new parameters, and the same head serves all three question types.")
P.h("3.3  Training", 2)
P.p("Each training example is one labelled question. The loss is soft cross-entropy against the target distribution; for "
    "~~score~~ questions it adds the ranked probability score, a proper scoring rule for ordered categories [@epstein], with "
    "weight 1. Option order is shuffled at every step (except for ordered levels) to remove position bias, and training steps "
    "are balanced 50/50 between text and image questions by oversampling the smaller modality. Examples longer than 2,048 "
    "tokens are skipped rather than truncated, and images are capped at 262,144 pixels in training. We use AdamW with a "
    "learning rate of 2e-4 for LoRA and 1e-3 for the head, a cosine schedule with 5% warm-up, gradient clipping at 1.0, an "
    "effective batch of 16 questions, one epoch, and bfloat16 autocast. The checkpoint with the lowest validation "
    "log-loss, evaluated every 100 steps, is kept (Appendix B).")
P.h("3.4  Baselines", 2)
P.bullets([
    "**B0, base model, zero-shot.** The same state, question and options, with options lettered A, B, C, ... and the model asked "
    "to answer with a letter; probabilities are the softmax of the next-token logits of the letters. It supports at most 26 "
    "options; questions beyond that count as wrong.",
    "**B1, base model with temperature scaling.** B0 with one temperature per question type fitted on the same validation set "
    "as our model's temperatures. This changes confidence but never the predicted option.",
    "**B2, letter-readout head.** Our framework (same LoRA, data, prompts, prefix cache, soft loss and per-type calibration), "
    "but the answer is read from the tied language-model logits of the letter tokens, as in SALSA-style readouts [@salsa], instead "
    "of the learned answer head. Like B0 it supports at most 26 options.",
])
P.h("3.5  Training mixes and inference options", 2)
P.p("Two training mixes are compared. The **narrow mix** has 12 public sources and 14,997 rows (Appendix C). The **broad mix** "
    "adds 12 further sources, for 26,413 rows and 31,059 questions. Entire task families are held out of both mixes: paraphrase "
    "detection, hateful, offensive and ironic tweets, financial-news sentiment, and digit, clothing and leaf-disease images. No "
    "Decision Index benchmark is used for training, and none of the broad mix's training rows overlapped the Decision Index "
    "suites by 13-gram text overlap (image hashes were not checked).")
P.p("Two optional inference techniques are studied. **Option-order averaging** answers each choice question under n option orders "
    "(the original, the reverse, and seeded shuffles) and averages the probabilities; yes/no and ordered questions are not "
    "permuted. A **seed ensemble** averages the temperature-calibrated probabilities of several independently trained models.")

# ====================================================================== 4 Setup
P.h("4  Experimental setup")
P.h("4.1  Test collections", 2)
P.p("Five test collections are used, none overlapping any training mix (Table 1). Three of them (16 open datasets, the two "
    "fresh datasets, the three zero-shot sources) and the many-option set were available while design choices were made. "
    "The **confirmation set** was built after the comparison was complete and influenced no decision; it is the cleanest estimate "
    "of performance on new datasets. Dataset sources and citations are in Appendix C.")
P.table("Test collections. All questions come from datasets that appear in no training mix.",
        ["Collection", "Questions", "Datasets", "Role"],
        [["16 open datasets", "3,678", "11 text, 5 image (SciQ, QASC, RTE, QNLI, MRPC, SNLI, IMDB, two hate and offensive-language tweet sets, financial news, 20 Newsgroups, CIFAR-10, CIFAR-100 superclasses, Fashion-MNIST, MNIST, bean leaves)", "built before the experiments; used to compare variants"],
         ["Fresh", "600", "ARC-Challenge, tweet irony", "used to compare variants"],
         ["Zero-shot", "720", "emotion, OpenBookQA, unseen Food-101 dishes", "used to compare variants"],
         ["Many options", "700", "CIFAR-100 (100 options), BANKING77 (77), GoEmotions (28)", "used to study the many-option weakness"],
         ["**Confirmation**", "2,706", "13 datasets: CB, COPA, WiC, four tweet tasks, Rotten Tomatoes, ARC-Easy, STL-10, GTSRB, RESISC45, SVHN", "built after the comparison; influenced no decision"]],
        [1.0, 0.75, 3.3, 1.45], size=8.5, left=(0, 2, 3))
P.h("4.2  Metrics and statistics", 2)
P.p("We report accuracy, expected calibration error (ECE, 15 equal-width bins) [@naeini] and negative log-likelihood (NLL, "
    "the log-loss of the probability given to the correct option). Accuracy is reported \"micro\" (all questions pooled) and "
    "\"macro\" (average over datasets). A question a model cannot answer, such as one with more options than a letter readout "
    "supports, counts as wrong with a fixed log-loss of ln(10^6).")
P.p("Uncertainty uses a paired, dataset-stratified bootstrap [@efron]: questions are resampled with replacement within each "
    "dataset (1,000 resamples) and the same resample is applied to every run, so differences between runs are paired and "
    "their intervals are much tighter than those of independent estimates. We report 95% percentile intervals and, for "
    "orientation, two-sided bootstrap p-values. Group comparisons average over three seeds per recipe and report the "
    "spread between seeds. We apply no correction for multiple comparisons, so the p-values are descriptive; claims "
    "rest on intervals and on consistency across collections. The released model is the seed with the lowest "
    "validation log-loss, a rule fixed before test results were examined.")
P.h("4.3  Implementation and compute", 2)
P.p("All experiments ran on one NVIDIA RTX 5070 Laptop GPU (8 GB). Training the broad mix takes 1,925 steps (about 2.6 hours, "
    "6.2 GB peak memory); the narrow mix takes 1,211 steps (about 1.6 hours).")

# ====================================================================== 5 Results
P.h("5  Results")
P.h("5.1  Fine-tuning against a calibrated base model (RQ1)", 2)
P.p("Table 2 and Figure 1 compare the released model (broad mix, seed chosen by validation log-loss) with the base model, "
    "with and without temperature scaling. Temperature scaling never changes the predicted option, so the two base columns have "
    "equal accuracy by construction.")
P.table("Accuracy (%) by collection. The last column is the paired difference between the released model and the base "
        "model with temperature scaling, with a 95% bootstrap interval.",
        ["Collection (questions)", "Base", "Base + temp. scaling", "Original mix", "Released", "Released minus base + TS"],
        [["16 open datasets (3,678)", "71.1", "71.1", "76.8", "78.0", "+6.9 [+5.5, +8.3]"],
         ["Confirmation, up to 26 options (2,306)", "69.2", "69.2", "70.9", "72.0", "+2.8 [+1.2, +4.4]"],
         ["Fresh (600)", "61.2", "61.2", "69.2", "76.2", "+15.0 [+11.2, +18.8]"],
         ["Zero-shot (720)", "68.3", "68.3", "67.5", "70.0", "+1.7 [-1.3, +4.3]"],
         ["Confirmation, all (2,706)", "58.9", "58.9", "68.4", "69.6", "+10.7*"],
         ["Many options, 27 to 100 (700)", "cannot answer", "cannot answer", "46.4", "48.4", "n/a"]],
        [2.15, 0.7, 0.9, 0.75, 0.7, 1.4], size=8.5,
        notes="\"Original mix\" is the narrow-mix model (one seed). *This row includes 400 questions with 43 or 45 options that "
              "the 26-letter baselines cannot answer, so the difference is inflated; the row above is the like-for-like estimate.")
P.figure(os.path.join(FIG, "fig1_gain_over_baseline.png"),
         "Accuracy gain over the base model with temperature scaling, with 95% paired bootstrap intervals. The gain is "
         "largest on the datasets closest to the training families (fresh, 16 open) and small on the confirmation datasets "
         "that influenced no decision; on the three zero-shot sources it is not distinguishable from zero.", 6.0)
P.p("The gain is real but depends strongly on the collection. It is +6.9 points on the 16 open datasets, which guided "
    "development, +2.8 on the confirmation datasets, and not significant on the zero-shot sources. A direct reading of the first "
    "row alone would overstate the benefit for a user with a new dataset; the confirmation row is the better guide. The gains "
    "concentrate in questions with few options (Section 5.5) and in tasks that resemble training families: offensive-language "
    "tweets (+36 points), financial-news sentiment (+34), SNLI (+22) and RTE (+11) in the open set (Appendix A).")
P.h("5.2  Calibration (RQ2)", 2)
P.table("Calibration. ECE and NLL (lower is better) for the raw base model (B0), the base model with temperature scaling (B1), "
        "the narrow-mix model and the released model.",
        ["Collection", "Metric", "Base (B0)", "Base + TS (B1)", "Original mix", "Released"],
        [["16 open datasets", "ECE", "0.240", "0.068", "0.037", "0.043"],
         ["", "NLL", "2.04", "0.80", "0.59", "0.58"],
         ["Confirmation, up to 26 options", "ECE", "0.255", "0.099", "0.040", "0.044"],
         ["", "NLL", "2.66", "0.96", "0.81", "0.81"],
         ["Fresh", "ECE", "0.357", "0.231", "0.070", "0.045"],
         ["", "NLL", "3.26", "0.82", "0.66", "0.56"],
         ["Zero-shot", "ECE", "0.266", "0.099", "0.054", "0.070"],
         ["", "NLL", "2.36", "0.92", "0.90", "0.91"]],
        [1.9, 0.6, 0.85, 1.0, 0.95, 0.8], size=8.5)
P.p("The raw base model is badly overconfident (ECE 0.24 to 0.36). A single temperature per question type, fitted on "
    "validation data, cuts its ECE by more than two thirds and its NLL by about 60% on the open datasets, with no change in accuracy. "
    "Measured against the gap between the raw base model and the released model, temperature scaling closes 87% of the ECE gap on "
    "the open datasets and 74% on the confirmation datasets. Fine-tuning retains a smaller edge (ECE 0.043 against 0.068 on the "
    "open datasets, NLL 0.58 against 0.80), which is statistically clear for NLL on every collection except zero-shot and for ECE "
    "on three of four, but it is not the large effect that comparing against the raw base model suggests (Figure 2).")
P.figure(os.path.join(FIG, "fig2_reliability.png"),
         "Reliability on the 16 open datasets: accuracy in each confidence bin (bins with at least 15 questions). The raw base "
         "model is far below the diagonal; temperature scaling moves it most of the way; the released model is closest.", 3.8)
P.h("5.3  Training on more tasks (RQ3)", 2)
P.p("Table 4 and Figure 3 compare three seeds of the narrow mix with three seeds of the broad mix. Because single seeds vary "
    "by one to three points (the original model's ECE of 0.037 on the open datasets was a favourable seed; the other two seeds "
    "scored 0.060 and 0.050), we average over seeds and bootstrap the difference of means.")
P.table("Narrow versus broad training mix, three seeds each: mean accuracy (%) with standard deviation across seeds, and "
        "the difference of means with a paired 95% bootstrap interval.",
        ["Collection", "Narrow mix", "Broad mix", "Difference"],
        [["16 open datasets", "76.2 ± 0.6", "77.6 ± 0.3", "+1.4 [+0.8, +2.0]"],
         ["Fresh", "68.8 ± 1.8", "73.3 ± 2.6", "+4.6 [+2.8, +6.5]"],
         ["Zero-shot", "68.8 ± 1.4", "70.4 ± 0.6", "+1.6 [+0.2, +3.2]"],
         ["Many options (27 to 100)", "45.2 ± 2.1", "50.3 ± 2.7", "+5.1 [+3.5, +6.8]"],
         ["Confirmation (13 datasets)", "68.7 ± 0.4", "70.1 ± 0.4", "+1.3 [+0.5, +2.1]"]],
        [2.2, 1.3, 1.3, 1.7], size=9,
        notes="Calibration error did not change consistently (differences in ECE were not significant except on the fresh "
              "datasets, where the broad mix was better by 0.049); log-loss improved on the open, fresh and many-option sets.")
P.figure(os.path.join(FIG, "fig4_breadth_three_seeds.png"),
         "Accuracy of three seeds of each training mix (dots) and their mean (bar), per collection. Numbers above the "
         "pairs are the difference of means in points.", 6.1)
P.p("The broad mix improves accuracy on every collection, including the confirmation set that influenced no decision, and "
    "the improvements exceed the spread between seeds on the fresh and many-option collections. This is consistent with "
    "the instruction-tuning literature, in which breadth of tasks drives generalisation [@wei,@sanh,@chung], at a scale of a "
    "few dozen sources rather than hundreds. We did not vary the number of tasks systematically, so we cannot say how the "
    "benefit scales.")
P.h("5.4  A learned answer head or a letter readout? (RQ4)", 2)
P.table("Learned answer head against the letter-readout head (B2), same data, prompts, LoRA, cache and calibration "
        "(narrow mix, one seed for the letter head).",
        ["", "16 open", "Fresh", "Zero-shot", "Confirmation (up to 26 options)", "Options above 26"],
        [["Accuracy, letter head minus answer head", "-0.2 [-1.4, +0.9]", "+0.2", "+4.4 [+1.8, +6.9]", "70.7 vs 71.1 (3-seed mean)", "0.0 vs 46.4"],
         ["ECE, answer head", "0.037", "0.070", "0.054", "0.047", "n/a"],
         ["ECE, letter head", "0.073", "0.108", "0.074", "0.073", "n/a"]],
        [1.8, 0.95, 0.5, 0.95, 1.3, 0.9], size=8.5)
P.p("The two heads are tied on accuracy on the open, fresh and confirmation collections. The letter readout is significantly "
    "better on the zero-shot sources, plausibly because it retains the base model's own mapping from options to letters, "
    "whereas the learned head is trained only on our mixes. The learned head is better calibrated and is the only one that can "
    "answer questions with more than 26 options, where the letter head and the base model score zero. If an application never "
    "exceeds 26 options, a letter-readout fine-tune is therefore a serious and simpler alternative.")
P.h("5.5  Number of options, and two remedies (RQ5)", 2)
P.figure(os.path.join(FIG, "fig3_by_option_count.png"),
         "Accuracy by number of options per question, pooled over the open, fresh and zero-shot collections (4,998 questions). "
         "Fine-tuning helps most on yes/no and few-option questions; above ten options the fine-tuned models are below the base "
         "model with temperature scaling, and averaging over four option orders recovers only part of the gap for the released model.", 6.2)
P.p("Splitting questions by number of options explains the aggregate numbers (Figure 4). On yes/no questions the released model "
    "gains 11 points over the temperature-scaled base model (77.4 against 66.1) and on three- and four-option questions 10 points "
    "(82.2 against 72.6); on five to ten options the gain shrinks to under two points; and above ten options the released "
    "model is 4.5 points ~~below~~ the base model (54.2 against 58.7). The training data contain few questions with many "
    "options, which is the likely cause, but we did not isolate it further.")
P.table("Remedies for the many-option weakness and other techniques. Differences are against the model they modify.",
        ["Technique", "Result"],
        [["Average over 4 option orders (released model)", "+1.0 [+0.4, +1.5] on the 16 open datasets; +0.4 [-0.4, +1.1] on the confirmation set; lower log-loss on four of five collections; options above ten: 54.2 to 56.9 (base model: 58.7). Median request latency rises from 69 ms to 70 to 93 ms (322 ms on 27 to 100 options)."],
         ["Average over 4 option orders (original mix, one seed)", "+1.3 [+0.7, +1.8] on the 16 open datasets; closes the gap above ten options (54.0 to 59.8)."],
         ["Three-seed ensemble (original mix)", "+0.8 [+0.1, +1.5] on the 16 open datasets, +1.9 [+0.3, +3.6] on zero-shot, at three times the inference cost."],
         ["Option-count augmentation (extra training copies padded to 12 to 40 options)", "**No benefit** over the broad mix alone: 27 to 100 options 52.4 against 53.4; 16 open datasets 77.6 against 77.3 (one seed each)."]],
        [2.3, 4.2], size=8.5, left=(0, 1))
P.p("Option-order averaging is a small but consistent improvement in log-loss and a significant accuracy gain on the 16 open "
    "datasets, but for the released model it does not close the many-option gap. Option-count augmentation, our attempt to "
    "fix the weakness through the data, did not help. We conclude that the weakness remains and that practitioners should not "
    "rely on the model, without further mitigation, for questions with many options.")
P.h("5.6  Efficiency", 2)
P.p("The median request takes 69 ms on the validation mix on the laptop GPU; a short single-question text request takes 39 ms, eight "
    "questions about one short text 89 ms, and an image with eight questions 434 ms. Encoding the state once and sharing it "
    "across questions is 3 to 6 times faster than re-reading it for each question when eight questions are asked about the "
    "same input (Appendix D). Peak memory is 6.2 GB in training and 7.4 GB in the latency benchmark with 4,000-token inputs.")

# ====================================================================== 6 Discussion
P.h("6  Discussion")
P.p("**What the results support.** Fine-tuning a 2B vision-language model into a decision model improves accuracy over a "
    "calibrated base model by roughly three points on new datasets, more on tasks close to the training families, and not "
    "at all on the zero-shot sources we tried. It also improves log-loss and, modestly, calibration error. Broader training "
    "is the clearest lever. The learned head is not needed for accuracy but is needed to exceed 26 options.")
P.p("**Practical guidance.** (i) Compare against a temperature-scaled base model before crediting fine-tuning with calibration. "
    "(ii) Prefer a broad training mix over a narrow one. (iii) If options never exceed 26, consider a letter-readout head. "
    "(iv) Treat questions with more than ten options with caution; option-order averaging helps somewhat. (v) Report intervals "
    "and seeds: single-seed differences of one to three points were common here and would have led to wrong conclusions.")
P.p("**Limitations and threats to validity.** The study uses one base model, one GPU, one LoRA configuration, and one "
    "epoch; larger models, longer training or full fine-tuning could behave differently. Three seeds were run for the main "
    "comparison but only one for the letter head, the augmentation run and the per-technique results. The first four test "
    "collections guided choices, so their gains are optimistic; only the confirmation set was untouched, and its gains are smaller. "
    "Contamination of the base model's pre-training data with the test datasets was not checked. Some test datasets are close to "
    "training tasks (SNLI and RTE resemble MNLI), so part of the gain there is transfer within a family; the held-out families "
    "(paraphrase, hateful and offensive tweets, financial sentiment, digits, clothing, leaf disease) are listed in Section 3.5. "
    "The decontamination check covered text only. We did not compare against other released decision models, which would be "
    "the most informative next experiment.")
P.p("**The Decision Index.** An earlier version of the model (narrow mix) was run through the Decision Index kit [@diKit] "
    "(Appendix E). The result, an index of 17.4, is not a leaderboard entry: the official suite could not be downloaded, so it "
    "was rebuilt from public sources with core rows that are not byte-identical to the official ones, and two retrieval "
    "benchmarks were not run because they cannot finish on an 8 GB GPU. The released model was not re-run, and the vision board "
    "was not run at all. These numbers should not be compared with leaderboard scores.")
P.p("**Ethics and broader impact.** A decision model that outputs well-calibrated probabilities can support triage and "
    "abstention, but its confidence is only as good as its validation data; the abstention thresholds of any application "
    "should be set on that application's own data. The model is weak on expert-level and long-context questions and was trained "
    "and evaluated on English-language, publicly available datasets under their own licences, so it should not be used for "
    "consequential decisions about people without further validation.")

# ====================================================================== 7 Conclusion
P.h("7  Conclusion")
P.p("We asked what fine-tuning buys a small vision-language decision model, and answered with calibrated baselines, paired "
    "intervals, three seeds and a confirmation set built after the comparison. The honest answer is a modest and "
    "task-dependent accuracy gain over a calibrated base model, a calibration advantage that is much smaller than it first "
    "appears, a clear benefit from training on more tasks, and a weakness on questions with many options that cheap remedies "
    "only partly address. We release everything needed to check or extend these results.")

# ====================================================================== statements
P.h("Reproducibility and availability", 2)
P.p("Code, training and evaluation scripts, test-set builders and the model recipe: github.com/agarwalastitwa027-svg/mini-jev-multimodal. "
    "Model weights (LoRA adapters and answer head, Apache-2.0) and model card: huggingface.co/AstitwaAgarwal/multimodal-jev. "
    "The per-question predictions of every run in this paper, from which all tables can be regenerated with the comparison "
    "tools, are in the Hugging Face dataset AstitwaAgarwal/decision-index-results (folder study/records). Public datasets "
    "are downloaded by the scripts under their own licences and are not redistributed.")
P.h("Use of AI assistance", 2)
P.p("The software, the experiments and drafts of this manuscript were produced with the assistance of an AI model (Claude, "
    "Anthropic) working under the author's direction. The author is responsible for the content. All numerical results come "
    "from the result files in the repository. The arXiv references in the related-work section were checked against their "
    "abstract pages; web sources are cited as accessed, and the dataset citations in Appendix C should be re-verified "
    "before submission.")
P.h("Independence", 2)
P.p("This is an independent study. It is not affiliated with, derived from or endorsed by TypeSafe AI or the maintainers of "
    "the Decision Index. Jev's architecture and weights are unpublished, and no claim is made to reproduce the product.")

# ====================================================================== references
P.h("References")
# references are listed after all citations (including those in the appendix) have been registered; see below

# ====================================================================== appendices
P.page_break()
P.h("Appendix A  Per-dataset results")
P.p("Accuracy (%), ECE and NLL for the base model, the base model with temperature scaling, and the released model. Chance is the "
    "average of 1/number of options. The base model cannot answer questions with more than 26 options (GTSRB and RESISC45 "
    "in the confirmation set), which count as wrong.")
for cap, name in (("16 open datasets.", "table_test_more.md"), ("Confirmation datasets (never used for any decision).", "table_test_confirm.md")):
    hdr, rows = md_table(os.path.join(ROOT, "results", "study", name))
    hdr = ["Dataset", "n", "Chance %", "Base acc", "Base+TS acc", "Released acc", "Base+TS ECE", "Released ECE", "Base+TS NLL", "Released NLL"]
    n = len(hdr)
    P.table(cap, hdr, rows, [1.25, 0.4, 0.55] + [0.62] * (n - 3), size=7.5, bold_last=True,
            notes="TS = temperature scaling. Base+TS has the same accuracy as Base by construction.")

P.h("Appendix B  Training and evaluation settings")
P.table("Hyperparameters of the released model (identical for the narrow mix except the number of steps).",
        ["Setting", "Value"],
        [["Base model", "Qwen3-VL-2B-Instruct, language-model head removed, vision tower frozen"],
         ["LoRA", "rank 16, alpha 32, dropout 0.05, all linear projections of the text decoder (17.4M LoRA parameters; the answer head adds 9.5M)"],
         ["Answer head", "LayerNorm, two 2-layer MLPs (GELU), 256-dimensional scoring space, one temperature per question type"],
         ["Optimiser", "AdamW, weight decay 0.01; learning rate 2e-4 (LoRA) and 1e-3 (head); cosine schedule, 5% warm-up; gradient clipping 1.0"],
         ["Batch and schedule", "4 questions x 4 accumulation steps; one epoch; 1,925 optimiser steps (broad mix), 1,211 (narrow mix)"],
         ["Loss", "soft cross-entropy; plus ranked probability score (weight 1) for ordered-level questions"],
         ["Sampling", "option order shuffled every step; 50% of steps text, 50% image (smaller modality oversampled)"],
         ["Limits in training", "examples above 2,048 tokens skipped; images capped at 262,144 pixels"],
         ["Precision", "bfloat16 autocast; trainable parameters in float32"],
         ["Checkpoint selection", "lowest validation log-loss, evaluated every 100 steps (up to 1,500 validation questions)"],
         ["Calibration", "temperature per question type fitted on the validation set by minimising log-loss"],
         ["Inference limits", "16,384 tokens, 255 options, 8 images; images up to 262,144 pixels in this study"],
         ["Seeds", "0, 1, 2 for each recipe; the released model is the broad-mix seed with the lowest validation log-loss (seed 1)"],
         ["Hardware", "one NVIDIA GeForce RTX 5070 Laptop GPU, 8 GB; 6.2 GB peak in training"]],
        [1.6, 4.9], size=8.5, left=(0, 1))

P.h("Appendix C  Datasets")
P.p("Training sources (narrow mix, then the additional sources of the broad mix) and test sources. Hugging Face dataset "
    "identifiers are given for the copies used; original citations are in the reference list where available.", size=10)
P.table("Training sources.",
        ["Source", "Hugging Face id", "Mix", "Reference"],
        [["AG News, DBpedia-14, Yelp", "fancyzhx/ag_news, fancyzhx/dbpedia_14, Yelp/yelp_review_full", "narrow", "[@zhang15]"],
         ["BoolQ", "google/boolq", "narrow", "[@boolq]"],
         ["MNLI, SST-2", "nyu-mll/glue (mnli), stanfordnlp/sst2", "narrow", "[@williams,@glue,@sst]"],
         ["CommonsenseQA", "tau/commonsense_qa", "narrow", "[@csqa]"],
         ["ScienceQA", "derek-thomas/ScienceQA", "narrow", "[@scienceqa]"],
         ["A-OKVQA, ChartQA", "HuggingFaceM4/A-OKVQA, HuggingFaceM4/ChartQA", "narrow", "[@aokvqa,@chartqa]"],
         ["Food-101 (seen classes)", "ethz/food101", "narrow", "[@food101]"],
         ["Procedural charts and shapes", "generated by data/make_images.py", "narrow", "this work"],
         ["CoLA, STS-B", "nyu-mll/glue (cola, stsb)", "broad", "[@cola,@stsb]"],
         ["RACE", "ehovy/race", "broad", "[@race]"],
         ["SMS spam", "ucirvine/sms_spam", "broad", "[@sms]"],
         ["Yahoo Answers topics, Amazon polarity", "community-datasets/yahoo_answers_topics, fancyzhx/amazon_polarity", "broad", "[@zhang15]"],
         ["QuaRTz", "allenai/quartz", "broad", "[@quartz]"],
         ["SST-5", "SetFit/sst5", "broad", "[@sst]"],
         ["Oxford-IIIT Pets, Oxford Flowers", "timm/oxford-iiit-pet, nelorth/oxford-flowers", "broad", "[@pets,@flowers]"],
         ["EuroSAT, DTD textures", "tanganke/eurosat, tanganke/dtd", "broad", "[@eurosat,@dtd]"]],
        [1.7, 2.7, 0.55, 1.55], size=8, left=(0, 1, 3))
P.table("Test sources.",
        ["Collection", "Datasets (Hugging Face id)", "Reference"],
        [["16 open", "SciQ (allenai/sciq), QASC (allenai/qasc), RTE, QNLI, MRPC (nyu-mll/glue), SNLI (stanfordnlp/snli), IMDB (stanfordnlp/imdb), TweetEval hate and offensive (cardiffnlp/tweet_eval), financial news sentiment (zeroshot/twitter-financial-news-sentiment), 20 Newsgroups (SetFit/20_newsgroups), CIFAR-10, CIFAR-100 superclasses (uoft-cs/cifar10, uoft-cs/cifar100), Fashion-MNIST (zalando-datasets/fashion_mnist), MNIST (ylecun/mnist), beans (AI-Lab-Makerere/beans)", "[@sciq,@qasc,@rte,@squad,@mrpc,@bowman,@imdb,@tweeteval,@lang,@cifar,@fashion,@mnist]"],
         ["Fresh", "ARC-Challenge (allenai/ai2_arc), TweetEval irony", "[@arc,@tweeteval]"],
         ["Zero-shot", "emotion (dair-ai/emotion), OpenBookQA (allenai/openbookqa), Food-101 unseen classes", "[@carer,@openbookqa,@food101]"],
         ["Many options", "CIFAR-100 fine labels, BANKING77, GoEmotions (google-research-datasets/go_emotions)", "[@cifar,@banking,@goemotions]"],
         ["Confirmation", "CB, COPA, WiC (aps/super_glue), TweetEval emotion, sentiment, stance and emoji, Rotten Tomatoes (cornell-movie-review-data/rotten_tomatoes), ARC-Easy, STL-10, GTSRB, RESISC45 (tanganke/*), SVHN (ufldl-stanford/svhn)", "[@superglue,@cb,@copa,@wic,@tweeteval,@rotten,@arc,@stl10,@gtsrb,@resisc,@svhn]"]],
        [1.0, 3.9, 1.6], size=8, left=(0, 1, 2))

P.h("Appendix D  Latency")
hdr, rows = md_table(os.path.join(ROOT, "results", "latency.md"))
P.table("Latency of the released model on the RTX 5070 Laptop GPU (milliseconds, median / 95th percentile), with the state "
        "encoded once and shared (cached) and with each question re-reading the state (uncached). Peak memory 7.43 GB.",
        hdr, rows, [1.7, 1.0, 1.9, 1.9], size=8.5)

P.h("Appendix E  Decision Index run of an earlier version (unofficial)")
P.p("An earlier version of the model, trained on the narrow mix, was run through the Decision Index reproduction kit (edition "
    "0.2.1) [@diKit]: 149,432 requests, all answered, with an index of 17.43 (chance-corrected, 0 is random guessing), a raw "
    "average of 36.8%, and median, mean and 95th-percentile latencies of 61, 242 and 771 ms. By area the skill scores were: tools "
    "and automation 25.5, language understanding 19.9, retrieval and classification 17.0, knowledge and reasoning 13.4, arts "
    "and human taste 7.9. As explained in Section 6 this is not a leaderboard entry, it concerns an earlier version rather than the "
    "released model, and ToolRet-retrieval (8,032 rows) and BRIGHT-retrieval (1,384 rows) were not run and count as wrong.", size=10)

# references last (all keys registered by now)
# insert the reference list at the position of the "References" heading: build it into a fresh section at the end and move
doc = P.doc
ref_heading = [p for p in doc.paragraphs if p.text == "References"][0]
tmp = P.doc.add_paragraph()  # anchor
P.reference_list()
refs_paras = doc.paragraphs[-len(P.order):]
body = ref_heading._p.getparent()
anchor = ref_heading._p
for para in refs_paras:
    body.remove(para._p)
    anchor.addnext(para._p)
    anchor = para._p
tmp._p.getparent().remove(tmp._p)

cp = P.doc.core_properties
cp.title = "What Does Fine-Tuning Buy a Small Vision-Language Decision Model? A controlled study against calibrated baselines"
cp.author = "Astitwa Agarwal"
cp.subject = "Calibrated typed decisions from a 2B vision-language model"
cp.keywords = "decision models; calibration; vision-language models; LoRA; temperature scaling"
out = os.path.join(HERE, "mini_jev_paper.docx")
P.save(out)
print("wrote", out, "| citations:", len(P.order), "| tables:", P.tables, "| figures:", P.figures)
