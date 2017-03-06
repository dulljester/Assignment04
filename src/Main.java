import java.io.*;
import java.nio.file.Paths;
import java.util.*;
import static java.lang.System.*;

public class Main {
    private static BufferedReader br;
    private static BufferedWriter bw;
    private static Scanner inp;
    static Classifier c;
    private static DataHolder dataHolder;
    private static File infile = null, outfile = null;

    enum Outcomes {
        TP(0,0), FP(0,1), FN(1,0), TN(1,1);
        final int byClassifier, trueValue;
        Outcomes( int b, int t ) {
            byClassifier = b;
            trueValue = t;
        }
        public static Outcomes which( int b, int t ) {
            for ( Outcomes o: values() )
                if ( o.byClassifier == b && o.trueValue == t )
                    return o;
            return null;
        }
    }

    public static void main(String... args) throws Exception {
        String filename = null;

        inp = new Scanner(System.in);

        do {
            out.printf("What is the name of the file containing your data?\n");
            filename = inp.next();
            infile = new File(Paths.get("").toAbsolutePath().toString() + "/" + filename);
            if ( !infile.exists() || infile.isDirectory() ) {
                out.printf(MyUtils.ANSI_RED+"[error] The supplied file is either non-existent, read-protected or is a directory."+MyUtils.ANSI_RESET+"\n");
                infile = null;
            }
        } while ( infile == null );
        out.printf(MyUtils.ANSI_GREEN+"[done]"+MyUtils.ANSI_RESET+" reading from %s\n\n",infile.toString());

        bw = new BufferedWriter(new PrintWriter(outfile = new File("./Rules.txt")));
        double perc = 0.99999999;
        System.setIn(new FileInputStream(new File(filename)));
        go(perc);
    }

    static void go( final double percent ) throws Exception {
        int t = -1;
        boolean flag;
        dataHolder = DataHolder.getInstance(br = new BufferedReader(new InputStreamReader(System.in)));
        Map<Integer, String> binaryAttributes = dataHolder.getAllBinaryAttributes();
        do {
            out.printf("Please choose an attribute (by number):\n");
            for (Map.Entry<Integer, String> entry : binaryAttributes.entrySet())
                out.printf("%8d: %s\n", entry.getKey(), entry.getValue());
            out.printf("\nAttribute: ");
            flag = false ;
            if (!inp.hasNextInt() || (flag = true) && (t = inp.nextInt()) <= 0 || t > binaryAttributes.size()) {
                if ( !flag ) inp.next();
                out.printf(MyUtils.ANSI_RED + "[error] please input a number in range 1-" + binaryAttributes.size() + MyUtils.ANSI_RESET + "\n");
                t = -1;
            }
        } while (t == -1);
        if (dataHolder.setTargetAttribute(binaryAttributes.get(t))) {
            out.printf("\n" + MyUtils.ANSI_GREEN + "[done]" + MyUtils.ANSI_RESET + " target attribute set to "+MyUtils.ASCII_BOLD+binaryAttributes.get(t)+MyUtils.ANSI_RESET+"\n\n");
            trainTheClassifier(percent);
            classify(percent);
            out.printf("The result is in the file "+MyUtils.ANSI_YELLOW_BACKGROUND+MyUtils.ANSI_BLUE+outfile.toPath().normalize().toAbsolutePath().toString()+MyUtils.ANSI_RESET+"\n");
            out.println(MyUtils.ANSI_GREEN+"*** Algorithm Finished ***"+ MyUtils.ANSI_RESET);
        } else {
            throw new RuntimeException("DataHolder could not set target variable");
        }
    }

    static void trainTheClassifier( final double percent ) {
        int i,k;
        final Set<Long> inc = new HashSet<>();
        if ( percent <= 0 || percent >= 1 )
            throw new IllegalArgumentException("percentage has to be in [0,1]");
        //int n = (int)(dataHolder.numUniqTuples()*percent);
        int n = dataHolder.numUniqTuples();
        Long []t = dataHolder.getUniqTuples(n), trainingData;
        c = new DecisionTree();
        //k = n/3;
        k = n;
        do {
            for ( c.trainOnData(t,k), inc.clear(), i = k; i < n; ++i )
                if ( c.getPrediction(t[i]) != dataHolder.getOutcome(t[i]) )
                    inc.add(t[i]);
            Arrays.sort(t, k, n, new Comparator<Long>() {
                @Override
                public int compare(Long o1, Long o2) {
                    boolean b1 = inc.contains(o1), b2 = inc.contains(o2);
                    if ( b1 == b2 ) return 0;
                    return b1?-1:1;
                }
            });
            for (;k < n && inc.contains(t[k]); ++k ) ;
        } while ( !inc.isEmpty() );
    }

    static void classify( final double perc ) {
        int n = dataHolder.numUniqTuples();
        int [][]v = new int[2][n];
        Long []t = dataHolder.getUniqTuples(n);
        Map<Outcomes,Integer> cnt = new HashMap<>();
        for ( Outcomes o: Outcomes.values() )
            cnt.put(o,0);
        for ( int i = 0; i < n; ++i ) {
            v[0][i] = dataHolder.getOutcome(t[i]);
            v[1][i] = c.getPrediction(t[i]);
            Outcomes o = Outcomes.which(v[0][i],v[1][i]);
            cnt.put(o,cnt.get(o)+1);
        }
        try {
            bw.write(c.toString()+"\n");
            bw.flush();
        } catch ( IOException io ) {
            out.println("[classify()]: "+io.getMessage());
            io.printStackTrace();
            throw new RuntimeException(io);
        }
        //out.printf("%% of data used in training = %.2f, Accuracy %.2f\n",perc*100,(cnt.get(Outcomes.TP)+cnt.get(Outcomes.TN)+0.0)/n);
    }
}

