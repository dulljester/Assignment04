import java.util.concurrent.ThreadLocalRandom;

import static java.lang.Math.*;

public class MyUtils {

    public static final String ANSI_RESET = "\u001B[0m";
    public static final String ANSI_BLACK = "\u001B[30m";
    public static final String ANSI_RED = "\u001B[31m";
    public static final String ANSI_GREEN = "\u001B[32m";
    public static final String ANSI_YELLOW = "\u001B[33m";
    public static final String ANSI_BLUE = "\u001B[34m";
    public static final String ANSI_PURPLE = "\u001B[35m";
    public static final String ANSI_CYAN = "\u001B[36m";
    public static final String ANSI_WHITE = "\u001B[37m";

    public static final String ANSI_BLACK_BACKGROUND = "\u001B[40m";
    public static final String ANSI_RED_BACKGROUND = "\u001B[41m";
    public static final String ANSI_GREEN_BACKGROUND = "\u001B[42m";
    public static final String ANSI_YELLOW_BACKGROUND = "\u001B[43m";
    public static final String ANSI_BLUE_BACKGROUND = "\u001B[44m";
    public static final String ANSI_PURPLE_BACKGROUND = "\u001B[45m";
    public static final String ANSI_CYAN_BACKGROUND = "\u001B[46m";
    public static final String ANSI_WHITE_BACKGROUND = "\u001B[47m";
    public static final String ASCII_BOLD = "\033[0;1m";

    public static double I( int zeros, int ones ) {
        if ( zeros == 0 || ones == 0 )
            return 0.00;
        double x = (zeros+0.00)/(zeros+ones), y = (ones+0.0)/(zeros+ones);
        return -x*log(x)-y*log(y);
    }
    public static long LSB( long u ) { return u&((~u)+1); }
    public static boolean isEmptyLine( String s ) {
        for ( Character ch: s.toCharArray() )
            if ( ch != ' ' )
                return false ;
        return true ;
    }
    public static int []which = new int[1<<20];
    public static long BIT( int k ) {
        return (1L<<k);
    }
    public static long MASK( int k ) {
        return BIT(k)-1L;
    }
    public static int who( long u ) {
        if ( which[2] == 0 )
            for ( int k = 0; k < 20; ++k )
                which[(int)BIT(k)] = k;
        else assert which[4] == 2;
        if ( u < (1L<<20) )
            return which[(int)u];
        if ( u < ( 1L<<40) )
            return 20+which[(int)(u>>20)];
        if ( u < (1L<<60) )
            return 40+which[(int)(u>>40)];
        return 60+which[(int)(u>>60)];
    }

    public static int randint(int i, int j) {
        return ThreadLocalRandom.current().nextInt(i,j+1);
    }

    public static boolean oneBitSet(long signature) {
        return (signature&(signature-1)) == 0;
    }

    public static double rndm() {
        return ThreadLocalRandom.current().nextDouble(1.00);
    }
}
