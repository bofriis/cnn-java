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
public class Definition {

    String type;
    public int num_neurons;
    String activation;
    int group_size;
    
    public int sx; // filter size in x, y dims
    public int sy;
    
    public int filters;
    public int stride;
    public int pad;
    
    public int in_depth; // depth of input volume
    public int in_sx;
    public int in_sy;
        
    public int out_depth;
    public int depth;
    public int out_sx;
    public int out_sy;
    
    int width;
    int height;
    
    String layer_type;
    
    double l1_decay_mul;
    double l2_decay_mul;
    
    public double bias_pref;
    
    double learnignRate = 0.01;
     
    public double drop_prob = 0.5; // Standard dropout-sandsynlighed (kan tilpasses)

    public Definition() {};
    
    public Definition(String type, int num_neurons) {
        this.type = type;
        this.num_neurons = num_neurons;
    }
    
    public Definition(int group_size, String type) {
        this.type = type;
        this.group_size = group_size;
    }
    
    public Definition(String type) {
        this.type = type;
    }
}
