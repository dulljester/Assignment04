import java.io.*;
import java.nio.file.Paths;
import java.text.NumberFormat;
import java.util.*;
import static java.lang.System.*;

public class Bonus {
    private static BufferedReader br;
    private static BufferedWriter bw;
    private static Scanner inp;
    static Classifier c;
    private static DataHolder dataHolder;
    private static File infile = null, outfile = null;
    private static String filename;

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
        filename = null;

        inp = new Scanner(System.in);

        out.printf("\n\n" + MyUtils.ASCII_BOLD+"Bonus Part: "+MyUtils.ANSI_RESET+MyUtils.ANSI_BLUE+ "Binary Classification tool\n\n"+MyUtils.ANSI_RESET);

        do {
            out.printf("What is the name of the file containing your training data?\n");
            filename = inp.next();
            infile = new File(Paths.get("").toAbsolutePath().toString() + "/" + filename);
            if ( !infile.exists() || infile.isDirectory() ) {
                out.printf(MyUtils.ANSI_RED+"[error] The supplied file is either non-existent, read-protected or is a directory."+MyUtils.ANSI_RESET+"\n");
                infile = null;
            }
        } while ( infile == null );
        out.printf(MyUtils.ANSI_GREEN+"[done]"+MyUtils.ANSI_RESET+" reading training data from %s\n\n",infile.toString());

        bw = new BufferedWriter(new PrintWriter(outfile = new File("./Predictions.txt")));
        double perc = 0.99;
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
            do {
                out.printf("What is the name of the file containing your input data?\n");
                filename = inp.next();
                infile = new File(Paths.get("").toAbsolutePath().toString() + "/" + filename);
                if ( !infile.exists() || infile.isDirectory() ) {
                    out.printf(MyUtils.ANSI_RED+"[error] The supplied file is either non-existent, read-protected or is a directory."+MyUtils.ANSI_RESET+"\n");
                    infile = null;
                }
            } while ( infile == null );
            out.printf(MyUtils.ANSI_GREEN+"[done]"+MyUtils.ANSI_RESET+" reading input data from %s\n\n",infile.toString());
            System.setIn(new FileInputStream(new File(filename)));
            classify();
            out.printf("The result is in the file "+MyUtils.ANSI_YELLOW_BACKGROUND+MyUtils.ANSI_BLUE+outfile.toPath().normalize().toAbsolutePath().toString()+MyUtils.ANSI_RESET+"\n");
            out.println(MyUtils.ANSI_GREEN+"*** Classification Finished ***"+ MyUtils.ANSI_RESET);
        } else {
            throw new RuntimeException("DataHolder could not set target variable");
        }
    }

    static void trainTheClassifier( final double percent ) {
        int i,k;
        final Set<Long> inc = new HashSet<>();
        if ( percent <= 0 || percent >= 1 )
            throw new IllegalArgumentException("percentage has to be in [0,1]");
        int n = (int)(dataHolder.numUniqTuples()*percent);
        Long []t = dataHolder.getUniqTuples(n), trainingData;
        c = new DecisionTree();
        k = n/3;
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

    static void classify() throws Exception {
        br = new BufferedReader(new InputStreamReader(System.in));
        NumberFormat nf = NumberFormat.getInstance();
        nf.setMaximumFractionDigits(2);
        nf.setMinimumFractionDigits(2);
        br.readLine(); // read the header
        Map<Outcomes,Integer> cnt = new HashMap<>();
        int []v = new int[2];
        for ( Outcomes o: Outcomes.values() )
            cnt.put(o,0);
        int tuples = 0;
        for ( String buff; (buff = br.readLine()) != null; ++tuples ) {
            if ( MyUtils.isEmptyLine(buff) ) continue ;
            Long t = dataHolder.mapRowToLong(buff);
            v[0] = dataHolder.getOutcome(t);
            v[1] = c.getPrediction(t);
            bw.write(buff+","+v[0]+","+v[1]+"\n");
            Outcomes o = Outcomes.which(v[0],v[1]); // TP, FN, TN, FP?
            cnt.put(o,cnt.get(o)+1);
        }
        bw.write("Accuracy "+nf.format((cnt.get(Outcomes.TP)+cnt.get(Outcomes.TN)+0.0)/tuples));
        bw.flush();
    }
}
