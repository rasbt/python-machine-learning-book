# Why do some people hate neural networks/deep learning?

I also know many people who make disrespectful remarks about neural networks in genaral. Personally, I find recurrent and convolutional neural networks truly beautiful. However, there is this popular saying: "that if al that you have is a hammer everything starts to look like a nail"

The math behind neural nets is probably a bit harder to understand, but I don't think they are really black boxes. I think a neural net is not more of a black box than standard techniques like kernel SVMs of random forests. Actually, I think it is easier to explain backpropagation than kernel methods.

However, I think people in the biosciences prefer "interpretable" results, e.g., decision trees where they can follow the "reasoning" step by step. Unarguably, random forests are better at solving the prediction task since you don't have to worry so much about overfitting or pruning your tree; at the same time, we lose some of this "interpretability." Although, I think this is not really true. Feature importance computed from e.g., extremely randomized trees might be even more useful than looking at a single decision tree.

Please note that I don't blame the bio-research field for this kind of thinking, they really try to solve different problems.
Assuming biologists want to know which functional groups of a ligand are "interacting" with residues in the protein binding site. Of course, the primary goal is often to a good agonist or antagonist (inhibitor or drug) in a million-compound database to solve a particular problem; in addition, they are also trying to "understand" and "explain" the results.
