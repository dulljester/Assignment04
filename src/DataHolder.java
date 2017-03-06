import java.io.*;
import java.util.*;

public class DataHolder {

    private static DataHolder instance = null;
    private Integer targetVar;
    private String targVarName;

    public static DataHolder getInstance( BufferedReader br ) {
        if ( instance != null )
            return instance;
        return instance = new DataHolder(br);
    }

    private Map<Integer,String> nameOfAttribute = new HashMap<>();
    private Map<Integer,Map<String,Integer>> m = new TreeMap<>();
    private Map<Integer,Map<Integer,String>> im = new TreeMap<>();
    private Map<Integer,List<String>> database = new HashMap<>();
    private Map<Long,Integer> cnt = new HashMap<>();
    private Map<Long,Set<Long>> extensions = new HashMap<>(), generators = new HashMap<>();
    private int []width, offset;
    private int N;
    private long []mask;
    private final static int SH = 22;

    public long getActualOutcome( Long t ) {
        return readAttribute(t,targetVar);
    }
    public int getOutcome( Long t ) {
        return (getActualOutcome(t)>>offset[targetVar])==1?0:1;
    }

    public long getSignature( long t ) {
        long u = 0;
        for ( int i = 0; i < getN(); ++i )
            if ( readAttribute(t,i) != 0 )
                u |= (1L<<i);
        return u;
    }
    public long removeSubset( long t, long _mask ) {
        if ( getN() < SH )
            return t&mask[(int)(~_mask&MyUtils.MASK(getN()))];
        for ( ;_mask > 0; _mask &= ~MyUtils.LSB(_mask) ) {
            int i = MyUtils.who(MyUtils.LSB(_mask));
            t &= ~(MyUtils.MASK(width[i]) << offset[i]);
        }
        return t;
    }
    public long extractSubset( Long t, long _mask ) {
        if ( getN() < SH )
            return t & mask[(int)_mask];
        return removeSubset(t, (~_mask)&MyUtils.MASK(N) );
    }

    public void addExtension( Long from, Long to ) {
        if ( !extensions.containsKey(from) )
            extensions.put(from,new HashSet<>());
        if ( !generators.containsKey(to) )
            generators.put(to,new HashSet<>());
        extensions.get(from).add(to);
        generators.get(to).add(from);
    }

    public Map<Long,Integer> retrieveAll() {
        return cnt;
    }

    public int getN() {
        return nameOfAttribute.size();
    }

    public int getDBSize() {
        return database.size();
    }

    public String getNameOfAttribute( int colId ) {
        return nameOfAttribute.get(colId);
    }
    public int getWeight( Long t ) {
        return cnt.containsKey(t)?cnt.get(t):0;
    }
    public void addWeight(Long c, int weight) {
        if ( !cnt.containsKey(c) )
            cnt.put(c,0);
        cnt.put(c,cnt.get(c)+weight);
    }
    public double getSupport( Long t ) {
        return (getWeight(t)+0.00)/getDBSize();
    }
    public String getFieldValueName( Long t, int idx ) {
        assert im.get(idx).containsKey(Integer.valueOf((int)readAttribute(t,idx)>>offset[idx]));
        return im.get(idx).get(Integer.valueOf((int)readAttribute(t,idx)>>offset[idx]));
    }
    public long readAttribute( Long t, int idx ) {
        if ( idx < 0 || idx >= getN() )
            throw new IllegalArgumentException("idx = "+idx);
        return ((t>>offset[idx])&MyUtils.MASK(width[idx])) << offset[idx];
    }

    public int cardinality( Long t ) {
        return Long.bitCount(getSignature(t));
    }

    private DataHolder(BufferedReader br ) {
        assert br != null;
        try {
            readData(br);
            mapDataToLong();
        } catch ( Exception e ) {
            System.out.println("[DataHolder constructor]: "+e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }

    private int getTopmostNonzero( Long t ) {
        int j;
        for ( j = N-1; j >= 0 && readAttribute(t,j) == 0; --j ) ;
        if ( j < 0 )
            throw new IllegalArgumentException();
        return j;
    }

    public long removeTopItem( Long t ) {
        int j = getTopmostNonzero(t);
        long res = t&~((MyUtils.MASK(width[j]))<<offset[j]);
        assert Long.bitCount(getSignature(res))+1 == Long.bitCount(getSignature(t)): t+ " and "+res+"\n";
        return res;
    }
    public long getTopItem( Long t ) {
        int j = getTopmostNonzero(t);
        return t&((MyUtils.MASK(width[j]))<<offset[j]);
    }

    private Long mapRowToLong( List<String> lst ) {
        long res = 0;
        assert lst.size() == getN();
        lst = lst;
        for ( int i = 0; i < lst.size(); ++i )
            res |= ((long)(m.get(i).get(lst.get(i))) << offset[i]);
        assert res != 0;
        return res;
    }
    private Long mapRowToLong( String ... lst ) {
        long res = 0;
        assert lst.length == getN();
        lst = lst;
        for ( int i = 0; i < lst.length; ++i )
            res |= ((long)(m.get(i).get(lst[i])) << offset[i]);
        assert res != 0;
        return res;
    }

    /*
     * the assumption is that the order of the columns is the same as in the training data set
     */
    public Long mapRowToLong( String s ) {
        return mapRowToLong(s.replaceFirst("^\\s+","").split("\\s+"));
    }

    private void mapDataToLong() {
        N = getN();
        width = new int[N];
        offset = new int[N+1];
        for ( Map.Entry<Integer,Map<String,Integer>> entry: m.entrySet() ) {
            for (;(1 << width[entry.getKey()]) < entry.getValue().size(); ++width[entry.getKey()]) ;
            if ( 0 == (entry.getValue().size()&(entry.getValue().size()-1)) )
                ++width[entry.getKey()];
        }
        assert m.size() == N;
        for ( int i = 1; i <= N; ++i )
            offset[i] = offset[i-1]+width[i-1];
        assert offset[N] <= 62;
        for ( Map.Entry<Integer,List<String>> entry: database.entrySet() ) {
            long key = mapRowToLong(entry.getValue());
            if ( cnt.containsKey(key) )
                cnt.put(key,cnt.get(key)+1);
            else cnt.put(key,1);
        }

        if ( getN() < SH ) {
            mask = new long[1 << getN()];
            for (long u = 0; u < (1 << N); ++u) {
                for (long v = u; v > 0; v &= ~MyUtils.LSB(v)) {
                    int i = MyUtils.who(MyUtils.LSB(v));
                    mask[(int) u] |= (MyUtils.MASK(width[i]) << offset[i]);
                }
            }
        }
    }

    private void removeNoise( Map<Integer,Map<String,Integer>> counts ) {
        for ( int j = 0; j < getN(); ++j ) {
            String majorityElement = MyUtils.getMajorityElement(counts.get(j));
            for ( int i = 0; i < database.size(); ++i )
                if ( database.get(i).get(j).equals("?") )
                    database.get(i).set(j,majorityElement);
        }
    }

    private void readData( BufferedReader br ) throws Exception {
        assert br != null ;
        String s = br.readLine(),t;
        Scanner scan = new Scanner(s);
        int i,j,k,n = 0;
        Map<Integer,Map<String,Integer>> counts = new HashMap<>();
        /*
         * converting original column names to lowercase: needed for bonus part training data/input data compatibility
         */
        for ( ;scan.hasNext(); im.put(n,new HashMap<>()), m.put(n,new HashMap<>()), nameOfAttribute.put(n++,scan.next().toLowerCase()) ) ;
        for ( i = 0; i < n; ++i )
            counts.put(i,new HashMap<>());
        for ( i = 0; (s = br.readLine()) != null; ++i ) {
            if ( MyUtils.isEmptyLine(s) ) { i--; continue ; }
            database.put(i,new ArrayList<>());
            for ( scan = new Scanner(s), j = 0; scan.hasNext(); ++j ) {
                t = scan.next().toLowerCase();
                if ( !t.equals("?") ) {
                    if ( counts.get(j).containsKey(t) )
                        counts.get(j).put(t,counts.get(j).get(t)+1);
                    else counts.get(j).put(t,1);
                }
                database.get(i).add(t);
                if ( !m.get(j).containsKey(t) ) {
                    k = m.get(j).size()+1;
                    m.get(j).put(t,k);
                    im.get(j).put(k,t);
                }
            }
        }
        removeNoise(counts);
    }

    public void addAll( Map<Long, Integer> cn ) {
        for ( Map.Entry<Long,Integer> entry: cn.entrySet() )
            if ( this.cnt.containsKey(entry.getKey()) )
                this.cnt.put(entry.getKey(),this.cnt.get(entry.getKey())+entry.getValue());
            else
                this.cnt.put(entry.getKey(),entry.getValue());
    }

    public String toStr( Long t ) {
        StringBuilder sb = new StringBuilder("{");
        int l = 0;
        for ( long mask = getSignature(t); mask > 0; mask &= ~MyUtils.LSB(mask) ) {
            int i = MyUtils.who(MyUtils.LSB(mask));
            if ( ++l > 1 ) sb.append(",");
            sb.append(getNameOfAttribute(i)+"="+getFieldValueName(t,i));
        }
        sb.append("}");
        return sb.toString();
    }

    public double getConfidence(Long lhs, Long rhs) {
        assert cnt.containsKey(lhs|rhs);
        assert cnt.containsKey(lhs);
        return (getWeight(lhs|rhs)+0.00)/getWeight(lhs);
    }

    public long extractComplement(Long aLong, long mask) {
        return extractSubset(aLong,(~mask)&MyUtils.MASK(getN()));
    }

    public boolean compatible(Long x, Long y) {
        return getTopmostNonzero(x) < getTopmostNonzero(y);
    }

    public int numUniqTuples() {
        return retrieveAll().keySet().size();
    }
    public Long[] getUniqTuples( int n ) {
        n = Math.min(n,numUniqTuples());
        Long []t = new Long[n];
        int k = 0;
        for ( Long x: cnt.keySet() ) {
            t[k++] = x;
            if ( k == n ) break ;
        }
        return t;
    }

    public Map<Integer,String> getAllBinaryAttributes() {
        Map<Integer,String> res = new TreeMap<>();
        int k = 0;
        for ( Map.Entry<Integer,Map<Integer,String>> entry: im.entrySet() )
            if ( entry.getValue().size() == 2 )
                res.put(++k,getNameOfAttribute(entry.getKey()));
        return res;
    }

    public boolean setTargetAttribute(String s) {
        for ( Map.Entry<Integer,String> entry: nameOfAttribute.entrySet() )
            if ( entry.getValue().equals(s) ) {
                targetVar = entry.getKey();
                targVarName = entry.getValue();
                return true ;
            }
        return false ;
    }

    public String getTargVarName() {
        return targVarName;
    }

    public String getTargClass(int value) {
        assert im.containsKey(targetVar);
        assert im.get(targetVar).containsKey(value): "value = "+value;
        return im.get(targetVar).get(value);
    }

    public int getTargVariable() {
        return targetVar;
    }
}

