# Is R used extensively today in data science?

"Extensively" is a relative term, so let me discuss this in comparison to other languages.
I would say that R was probably THE language for doing statistics or "data science" work about 5-10 years ago. Today, as the Python sci-stack caught up and keeps growing, it's about as widely used as Python for similar tasks. I can see a shift more towards Python in future though because there seems to be more development going on at the moment towards scalability and computational efficiency. For example,

- Blaze for out-of-core analysis of big datasets
- Dask for parallel computing on multi-core machines or on a distributed clusters
- Theano and Tensorflow for the optimization and evaluation of mathematical

expressions involving multi-dimensional arrays utilizing GPUs
and many, many more. Although R is fine for "small scale" analyses, performance can be (become) a big weakness of R for real-world applications.
However, keep in mind that Scala is also big on the rise right now, take Spark for instance.
Eventually, I think it all depends on the task and the problem you'd want to solve. For "smallish" analysis and projects, Python's default sci-stack and R work just fine. For large-scale distributed computing, you'd typically use Spark (written in Scala). For deep learning, you use Theano or Tensorflow (via Python) or Torch (written in Lua).

(If all you have is a hammer, everything looks like a nail :).)
