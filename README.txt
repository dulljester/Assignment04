This is an implementation of ID3 algorithm, following the original paper http://dl.acm.org/citation.cfm?id=637969
GitHub page: https://github.com/dulljester/Assignment04

FIXME:
    data2: accuracy 50%

--Compile & run

    compile: make
    run: java Main
        NOTE: for simplicity, you can put the dataset inside the directory with the sources (this way you don't have to provide full path)


        ***** BREAKDOWN BY EVALUATION CRITERIA *****


--Introduction to the code structure / architecture

        * loadDB(): DataHolder class, private void loadDB( BufferedReader br ) throws Exception ;
        * buildFirstItemset(): AprioriTid class, private void buildFirstItemset() ;
        * pruneItemset(): AprioriTid class, public Map<Integer,Set<Long>> findAllLargeItemsets() ;
        * generateCandidates(): AprioriTid class, public Map<Integer,Set<Long>> findAllLargeItemsets() ;
        * createRules(): inside Main class, public static void main( String ... args ) ;
        * outputRules(): inside Main class, public static void main( String ... args ) ;

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

        Some "embarrassingly-parallel" tasks have been realized using Java threads -- via implementing the Callable interface as in the case of
        Joiner class (used for generating candidate itemsets from L_{k-1} x L_{k-1} join),
        or extending Java SE7's RecursiveTask as in the case of RulesMiner class.
        The datasets being small, this step may not look necessary, but it points to parallelization opportunities, so I though I might just as well add it.
        If Joiner class is made to extend Thread/implement Runnable, managing memory with low-level start/join calls is harder: with support = 0.1
        and confidence = 0.1 values as many as ~9200 rules are generated for sample data3, and on bluenose the program may run out of threads.
        That is why Java SE7's ExecutorService API is used for managing threads.
        To confine ourselves to meaningful rules only, we require support >= 0.05 and confidence >= 0.05 rather than 0.00

--Brief Description about bonus part

        I follow the original paper "Fast Algorithms for Mining Association Rules in Large Databases" by Rakesh Agrawal and Ramakrishnan Srikant
        They suggest that we can dispense with DB for support counting in subsequent iterations if we maintain a collection of frequent (k-1)-itemsets
        that reside within a certain transaction; the algorithm is called AprioriTid (hence my Java class). This is because not all transactions may
        carry large itemsets; we should keep only those that do; even this list will be further refined as the transactions with no k-large itemsets
        are dropped at subsequent iterations
        Inside AprioriTid class, there is a private class CandidateItemsetsTID -- essentially a list of frequent $k-1$-itemsets tagged by an ID --
        which is the transaction itself, because of our encoding (see above). All in all, the idea is implemented in
                AprioriTid :: public Map<Integer,Set<Long>> findAllLargeItemsets() ;


--Code Design

    --Modularity/Functionality:

        * MyUtils: utility collection of general-purpose operations and constants
        * DataHolder: interpets the long as a transaction, encodes a transaction as a long, provides access to transaction's fields, loads DB, etc.

        [reusability] I will use the above two classes for Assignments 4 & 5, too, as long as the dataset format remain the same.

        * AprioriTid: implementation of AprioriTid algorithm, yields the list of all the large itemsets (i.e. L_k for all k >= 2)
        * AssociationRule: wrapper around association rule with overridden toString() method for pretty-printing
        * RulesMiner: produces the actual association rules given the list of all large itemsets (L_k for all k >= 2, that is)
        * Joiner: extends Thread for multithreading during Joins (needed for L_{k-1} x L_{k-1} join)
        * Main: driver; interacts with the user, I/O set-up, etc.

    --Code readability and comments

        All the methods bear self-explanatory names and are supplied with asserts to check for pre/post conditions




