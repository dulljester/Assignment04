/**
 * Created by sj on 16/02/17.
 */
public interface Classifier {
    void trainOnData( final Long[]trainingData, final int len ) ;
    int getPrediction( Long t ) ;
}
