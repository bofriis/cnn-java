/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package rocks.propeller.karpathy;

/**
 *
 * @author gola
 */
public class ParamsAndGrads {
    public double[] params;
    public double[] grads;
    public double[] gsum; // momentum accumulator
    public double l1_decay_mul;
    public double l2_decay_mul;

    public ParamsAndGrads(double[] params, double[] grads, double l1, double l2) {
        this.params = params;
        this.grads = grads;
        this.l1_decay_mul = l1;
        this.l2_decay_mul = l2;
        this.gsum = new double[params.length];
    }
}
