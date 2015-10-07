The *Known Errors* Leaderboard
========================

I tried my best to cut all the little typos, errors, and formatting bugs that slipped through the copy editing stage. Even so, I think it is just human to have a little typo here and there in a first edition. I know that this can be annoying as a reader, and I was thinking to associate it with something positive. Let's have a little leaderboard (inspired by Donald Knuth's "[Known Errors in My Books](http://www-cs-faculty.stanford.edu/~uno/books.html)").

Every error that is not listed in the *Errata* yet will be rewarded with $0.25. I am a man of my word, but I want to be a little bit careful with the initial amount; I will probably increase it soon depending on how it goes.

The only catch here is that you won't receive any single cent. Instead, I am going to donate the amount to [UNICEF USA](http://www.unicefusa.org), the US branch of the United Nations agency for raising funds to provide emergency food and healthcare for children in developing countries.

I would be happy if you just write me a short [mail](mailto:mail@sebastianraschka.com) including the error and page number, and please let me know if you want to be listed on the public leaderboard.

## Donations

- Current amount for the next donation: $2.25
- Amount donated to charity: $0.00

## Leaderboard

1. Joseph Gordon ($0.75)
2. T.S. Jayram ($0.50)
3. S.R. ($0.50)
4. Ryan S. ($0.25)
5. Elias R. ($0.25)


...

# Errata

### Technical Notes

I am really sorry about you seeing many typos up in the equations so far. Unfortunately, the layout team needed to retype the equations for compatibility reasons. There were a lot of typos introduced during this process, and I tried my very best to eliminate all of these by comparing the pre-finals against my draft. Cross-comparing 450 pages was a tedious task, and it appears that several typos slipped through, so if you see something that does not make quite sense to you, please let me know.   



**Chapter 2**

- p. 20: Duplication error: Since we brought &theta; to the left, it should be "1 if z &#8805; 0" and not  "1 if z &#8805; &theta;" anymore (Joseph Gordon)
- p. 22:  *y<sup>(i)</sup>* is set to *1<sup>(i)</sup>*, rather it should be just *1* (T.S. Jayram)
- p. 25:  The link to `matplotlib` is misspelled, it should be http://matplotlib.org/users/beginner.html instead of http://matplotlib.org/ussers/beginner.html (T.S. Jayram)
- p. 35:  In the context "... update the weights by taking a step away from the gradient &nabla; J(w) ... weight change  &Delta;w defined as the negative gradient multiplied by the earning rate &eta; : &Delta; w = −&eta;&Delta;J(w)."  --> "&Delta;w = −&eta;&Delta;J(w)" should be "&Delta;w = −&eta;&nabla;J(w)" (Ryan S.)

**Chapter 3**

- p. 60: There is a missing product operator that defines the *likelihood* *L(**w**)* = ...=&prod;<sup>n</sup><sub>i=1</sub> (&phi;(z)<sup>(i)</sup>)<sup>y<sup>(i)</sup></sup> ... (Elias R.)

**Chapter 5**

- p.144: I wrote in the Linear Discrimnant section that "Those who are a little more familiar with linear algebra may know that the rank of the d×d-dimensional covariance matrix can be at most *d − 1* ..." Sorry, this is a little bit out of context. First of all, this is only true if *d >> N* (where *d* is the number of dimensions and *N* is the number of samples), and this should have been in the Principal Component Analysis section. Secondly, in context of the Linear Discriminant Analysis, the number of linear discriminants is at most <em>c-1</em> where <em>c</em> is the number of class labels; the in-between class scatter matrix <em>S<sub>B</sub></em> is the sum of <em>c</em> matrices with rank 1 or less.</strong> (S.R.)

### Language

**Preface**

- p. x: the phrase "--whether you want start from..." should be  "--whether you want to start from..." (Joseph Gordon)

**Chapter 2**

- p. 19: there should be a period between "otherwise" and "in" (this is towards the end of the page) (Joseph Gordon)

**Chapter 3**

- p. 89: The *petal with* near the bottom of this page should of course be *petal width* (S.R.)
