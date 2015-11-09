# Why did it take so long for deep networks to be invented?

It's not that "deep networks" haven't been around in the 1960s, but the problem was how to train them. In the 1970s, backpropagation was "invented" or re-discovered -- I don't want to quote a single resource here not to offend any of the parties involved since this is a sensitive topic those days ... In any case, the problem was the "vanishing gradient," when gradient-based methods were used for learning the weights. It was observed that there was no gain going beyond 1-3 hidden layers.


So back to the question: Deep network structures existed, but it was hard/impossible to train them appropriately. I'd say the 2 main reasons why this field experienced such a leap in the recent years
are


1. availability of computing resources
2. clever ideas to pre-train a neural network


The second point is what deep learning is all about; in a nutshell, we pre-train our deep neural networks using unsupervised learning, but this goes beyond the scope of the question ...   
