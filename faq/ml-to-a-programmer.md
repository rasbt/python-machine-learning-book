# How would you explain machine learning to a software engineer?


Software engineering is about developing programs or tools to automate tasks. Instead of "doing things manually," we write programs; a program is basically just a machine-readable set of instructions that can be executed by a computer.
Let's consider a classic example: e-mail spam filtering. Assuming that we have access to the source code of our e-mail client and know how to handle it, we could come up with an instinctive set of rules that may help us with our spam problem.


For example:
if not "sender in contacts":
if "subject line contains BUY!:
e-mail spam folder:"
else if ...


It is intuitive to say that coming up with these rules is a pretty tedious task. Needless to say that we have to test our spam filter on real-world data, evaluate and improve it constantly, change and update rules, and so forth. Again, our goal is automation: we want to write a set of instructions that automatically filters out spam e-mails so that we don't have to "manually" delete them from our e-mail inbox.

Now, **Machine learning is all about automating automation**! Instead of coming up with the rules to automate a task such as e-mail spam filtering ourselves, we **feed data to a machine learning algorithm, which figures out these rules all by itself.** . In this context, "data" shall be representative sample of the problem we want to solve -- for example, a set of spam and non-spam e-mails so that the machine learning algorithm can "learn from experience."


In "conventional" programming, we code up a set of rules, feed it to the computer together with the data, and hope that it produces the desired results.


**traditional programming:**


- set of rules + data -> computer -> results


In machine learning, we take data (e.g., e-mails), provide information about the desired results (spam and non-spam labels for these e-mails), and feed it to a learning algorithm, which in turn executed by a computer. The computer then *learns* a set of rules that we can use to automate (solve) our problem task.


**machine learning:**

- results + data -> machine learning algorithm + computer -> set of rules


**Or in other words, machine learning is about finding the optimal instructions to automate a task. Machine learning algorithms are instructions for computers to learn other instructions automatically from data or experience. Therefore, machine learning is the automation of automation.**   
