This is an implementation of ID3 algorithm, following the original paper http://dl.acm.org/citation.cfm?id=637969
GitHub page of the project: https://github.com/dulljester/Assignment04

--Compile & run

    compile: make
    run: java Main (or java Bonus)
        NOTE: for simplicity, you can put the dataset inside the directory with the sources (this way you don't have to provide full path)

        ***** BREAKDOWN BY EVALUATION CRITERIA *****

--Introduction to the code structure / architecture

        * readdata(): DataHolder class, private void readData( BufferedReader br ) throws Exception ;
        * getattrib(): Main/Bonus class, interaction with the user, I/O, determining the target attribute, etc.
        * maketree(): DecisionTree class implementing Classifier interface, the Node inner class constructor takes care of "maketree" functionality
        * getInformationGain(): determineSplittingVarIdx() method in DecisionTree class
        * doSplit(): determineSplittingVarIdx() and Node's constructor
        * printtree(): DecisionTree class has an overridden toString() method that calls a recursive "printMyself()" method of the Node class

--Specify limitations of the program( if any)

    --Itemset Encoding:

        I use a *long integer* to encode an itemset/transaction. More specifically, assume, for ease of exposition, just two columns
        in the data file. Assume that the first column takes on M values, and the second column takes on N different values.
        Then I would compute m = log2(M) and store an itemset (x,y) as a long integer (x|(y<<m)).
        Each of the different M values are represented by 1..M, with 0 meaning that the item is missing altogether.
        This way, as long as SUM(log_2(V_j)) <= 62, where V_j is the cardinality of the j-th attribute, we can safely
        encode an itemset in Java's built-in long primitive type, or indeed its wrapper class Long, to use with Collections --
        but we don't have to worry about this, since Java takes care of boxing/unboxing.

        This binary-decimal encoding, in addition to being pretty lightweight, allows us to take advantage of bit-parallelism.
        For example, enumerating all subsets of the given itemset is nothing but an enumeration of all the submasks of a given
        bit pattern we will call "signature" -- an integer with bits set to 1 only for positions for which our itemset has
        an entry. Technically, enumerating all the submasks of a bitpattern "base" is accomplished by a simple loop, like so

                    for ( int submask = base; submask > 0; submask = (submask-1) & base )

        Enumerating individual items is accomplished in the number of bits set in "signature" by repeatedly removing a least significant bit,
        like so

                    LSB(x) := (x & (~x+1))

        In short, we harness the power of bitwise operations to facilitate our access to data; all this (and additional) functionality
        is provided by the DataHolder class, which implements a Singleton design pattern (as DAOs, Data Access Objects, in many cases, do).
        Compound bitwise operations such as ones above are implemented in MyUtils class as static members.

        As to incapsulation, all the entities of the program are aware that an itemset is a Long. It is only that DataHolder alone
        knows how to interpret it, and each entity uses its own reference the DataHolder (unique!) object to extract information
        from an itemset (or transaction).

--Other instructions / comments to the user

        During the recursive process, when we run out of attributes but the classes are still mixed, we give the class label with "probability",
        which is essentially the fraction of 0-labels. If it is 50/50, we look at the parent node.

--Code Design

    --General Flow of the Algorithm:
        The Node instance is given a certain bitmask we'll call "signature", where set bits denote the attributes we can perform a split.
        The Node also has a "left" and "right" indices of a portion of training data that has been assigned to it (this way, we address a "globally"
        allocated training data and don't have to copy the data for each node)
        For each possible splitting variable, we sort the left-to-right portion of the data with respect to that attribute
        (here just-in-time anonymous classes of Java come in handy), which breaks the left-to-right portion into segments
        ready to be passed to the current node's children (if we decide to split on this attribute, that is).

    --Modularity/Functionality:

        * MyUtils: utility collection of general-purpose operations and constants
        * DataHolder: interprets the long as a transaction, encodes a transaction as a long, provides access to transaction's fields, loads DB, etc.

        [reusability] I have used the above classes for Assignment 3; and plan to do so for Assignments 5, too, as long as the dataset format remains the same.

        * Classifier: an interface with two methods -- trainOnData() and getPrediction();
          With this "coding-for-interface" pattern, for Assignment 5, all we have to do is to implement a NaiveBayes class
          that implements Classifier; all the other classes access the DecisionTree via the Classifier interface;
          the whole system is loosely coupled to the DecisionTree class. All we do is just plug-in another implementation
          of the Classifier interface.
        * DecisionTree implements Classifier: essentially, its private inner class Node does all the non-trivial work
        * Main: driver; interacts with the user, I/O set-up, etc.
        * Bonus: derives a decision tree from training data and classifies another dataset, reports accuracy and class labels

    --Code readability and comments

        All the methods bear self-explanatory names and are supplied with asserts to check for pre/post conditions
