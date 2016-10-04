package org.allenai.ml.sequences.crf;

import org.allenai.ml.linalg.Vector;
import org.allenai.ml.sequences.ForwardBackwards;
import org.allenai.ml.sequences.SequenceTagger;
import lombok.RequiredArgsConstructor;
import lombok.Setter;
import lombok.val;
import org.allenai.ml.util.Indexer;

import java.util.ArrayList;
import java.util.List;

@RequiredArgsConstructor
public class CRFModel<S, O, F extends Comparable<F>> implements SequenceTagger<S, O> {
    public final CRFFeatureEncoder<S, O, F> featureEncoder;
    public final CRFWeightsEncoder<S> weightsEncoder;
    // This is private because it's mutable. The weights() method
    // will return a copy
    private  final Vector weights;
    @Setter
    private InferenceMode inferenceMode = InferenceMode.VITERBI;

    public static enum InferenceMode {
        VITERBI,
        MAX_TOKEN
    }

    @Override
    public boolean equals(Object o) {
        val r = innerEquals(o);
        if(!r)
            System.err.println(this.getClass().getName() + " not equals");
        else
            System.err.println(this.getClass().getName() + " equals! Yay!");
        return r;
    }

    private boolean innerEquals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        CRFModel<?, ?, ?> crfModel = (CRFModel<?, ?, ?>) o;

        if (!featureEncoder.equals(crfModel.featureEncoder)) return false;
        if (!weightsEncoder.equals(crfModel.weightsEncoder)) return false;
        if (!weights.equals(crfModel.weights)) return false;
        return inferenceMode == crfModel.inferenceMode;
    }

    @Override
    public int hashCode() {
        int result = featureEncoder.hashCode();
        result = 31 * result + weightsEncoder.hashCode();
        result = 31 * result + weights.hashCode();
        result = 31 * result + inferenceMode.hashCode();
        return result;
    }

    /**
     * Return copy of weights
     */
    public Vector weights() {
        return weights.copy();
    }

    @Override
    public List<S> bestGuess(List<O> input) {
        if (input.size() < 2) {
            throw new IllegalArgumentException("Need to have at least two elements");
        }
        if (input.size() == 2) {
            // only have start stop, so return empty (unpadded)
            return new ArrayList<>();
        }
        input = new ArrayList<>(input);
        val indexedExample = featureEncoder.indexedExample(input);
        double[][] potentials = weightsEncoder.fillPotentials(weights, indexedExample);
        val forwardBackwards = new ForwardBackwards<>(featureEncoder.stateSpace);
        ForwardBackwards.Result fbResult = forwardBackwards.compute(potentials);
        if (inferenceMode == InferenceMode.VITERBI) {
            return fbResult.getViterbi();
        }
        double[][] edgeMarginals = fbResult.getEdgeMarginals();
        return forwardBackwards.compute(edgeMarginals).getViterbi();
    }
}
