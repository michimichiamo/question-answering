### 

### Limitations and Future Work

Although we reached good enough performances, we encountered several **limitations** that posed **constraints** on the quality of our results. We can identify various kinds of issues, in particular related to the **available resources** and to the **design choices**.

[Limitations:]

- For instance, we employed a **small** Transformers architecture and set a relatively low batch size (of 128) in order to comply with the computational resources limits. Likewise, the **time** required to perform the tests allowed us for poor hyperparameter tuning.

- As for the design choices, our approach required **specific preprocessing** operations such as the necessity to **split** tokenized input sequences to comply with the Transformers architecture. Moreover, we decided to **include** the resulting splits (**not containing** the answer), instead of discarding them.

- **Multiple answers** are another possible source of error, as they tend to confuse the network because of the **strong assumptions on the probability distribution** of samples due to the use of **Cross-Entropy as a loss function** (in particular, a univariate Gaussian). Actually, this is not ideal to model **multivalued** targets, since it produces a **Regression to the Mean**, while the average of several correct target values is **not necessarily** itself a correct value.
  Nevertheless, we noticed that in the majority of cases answers to the same question are **not mutually exclusive**, in particular either one span inside of the other, or sharing a piece of text. These cases tend to **behave relatively well,** since they mainly tend to **add noise** to the predictions.
  Still, the number of samples provided with multiple answers are a **scarce 0.3% of the total**, thus arguably having **little to no impact** on the actual performance of the network.

[Possible improvements:]

- Beyond the obvious **enhancement of computational resources**, possible improvements of our approach include the explicit **provision of unanswerable questions** within the dataset and the **support for multiple answers**.

- Although our model takes the former into account, they only represent about **0.1% of the samples**, which results in **great underrepresentation** of this situation. SQuAD v.2.0, on the other hand, provides about 50000 samples of unanswerable questions in addition to SQuAD v.1.1, which results in a **good balancing** and should allow for effective training.

- On the other hand, the problem of **multiple answers** could be solved for example by modelling the **posterior probability distribution** of the target data, conditioned on the input, which is addressed in **Mixture Density Networks**.

## Question Generation

### Outline

[Goal:]

- The goal of the Question Generation task is to develop a model that receives **a paragraph as input, along with an answer**, and is supposed to **generate a question** relevant to the paragraph and having the provided answer as actual answer.

- This task can be framed within the **Natural Language Generation** field, in particular belonging to the **Answer-Aware** (as opposed to Answer-Agnostic) Question Generation area.

[Method:]

- Since Neural QG recent approaches make extensive use of **Transformers**, we decided to rely on **T5** for our approach, which I am going to briefly describe in a few seconds.
  As Alessandro is going to mention, quantitative evaluation is performed through two very common metrics (**BLEU** and **METEOR**), although **many issues are known**, especially when they are deployed on Text Generation tasks.

### Background

Let's now take a look at the actual implementation of T5.

- According to the authors, it closely follows the original Transformers architecture with little modifications.

- In particular, a simplified version of Layer Normalization (without bias) is applied before (as opposed to after) the residual connection. As for the positional embeddings, a relative (as opposed to sinusoidal, absolute) encoding is used.

- The authors provide different versions of T5, depending on the size of the model: we exploited t5-small, in which both the encoder and the decoder consist each of 6 "blocks" (that is, self-attention and feed-forward Neural Network for the encoder, with an extra cross-attention for the decoder), while the self-attention consists of 8 heads, for an overall size of 60 million parameters. In the original implementation both the blocks and the attention heads are 12, for 220 million parameters overall.
  The self-attention mechanism in the decoder uses a form of autoregressive or causal self-attention, which only allows the model to attend to past outputs.

### Data preprocessing

The preprocessing phase is **very similar** to the Question Answering part.

- The main difference relies in the **tokenization**, since the input of the network consists of the **answer and the context**, while the expected ouput is the question. This also means that, due to the generative nature of the task (as opposed to classification), we need to tokenize also **target questions**.

- In addition, we **removed** samples for which the anwer is **not contained** within the context, as the answer is **needed** as **part of the input** to the network.

### Data preprocessing - Example

As we can see in the example, the T5 tokenizer is **very similar** to the DistilBERT one, just with different encoding **schemas**.
