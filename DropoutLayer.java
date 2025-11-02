package rocks.propeller.karpathy;

import java.util.Random;

import com.improaim.org.json.JSONException;
import com.improaim.org.json.JSONObject;

public class DropoutLayer implements ILayer {
    
    int out_sx;
    int out_sy;
    int out_depth;
    String layer_type = "dropout";
    double drop_prob;
    boolean[] dropped;

    Vol in_act;
    Vol out_act;

    Random random = new Random();

    public DropoutLayer(Definition opt) {
        this.out_sx = opt.in_sx;
        this.out_sy = opt.in_sy;
        this.out_depth = opt.in_depth;
        this.drop_prob = opt.drop_prob;
        this.dropped = new boolean[out_sx * out_sy * out_depth];
    }

    @Override
    public Vol forward(Vol V, boolean is_training) {
        this.in_act = V;
        this.out_act = V.cloneVol();

        if (is_training) {
            for (int i = 0; i < V.w.length; i++) {
                if (random.nextDouble() < drop_prob) {
                    out_act.w[i] = 0;
                    dropped[i] = true;
                } else {
                    dropped[i] = false;
                    out_act.w[i] /= (1.0 - drop_prob); // Scale op (inverted dropout)
                }
            }
        }

        return out_act;
    }

    @Override
    public void backward() {
        Vol V = this.in_act;
        Vol chain_grad = this.out_act;

        for (int i = 0; i < V.w.length; i++) {
            if (dropped[i]) {
                V.dw[i] = 0;
            } else {
                V.dw[i] += chain_grad.dw[i] / (1.0 - drop_prob); // Inverted dropout
            }
        }
    }

    @Override
    public ParamsAndGrads[] getParamsAndGrads() {
        return new ParamsAndGrads[0]; // Dropout har ingen parametre
    }

    @Override
    public int get_out_sx() { return out_sx; }
    @Override
    public int get_out_sy() { return out_sy; }
    @Override
    public int get_out_depth() { return out_depth; }
    @Override
    public String get_layer_type() { return layer_type; }
    @Override
    public Vol get_out_act() { return out_act; }
    @Override
    public void set_in_act(Vol v) {
        this.in_act = v;
    }

    @Override
    public Vol get_in_act() {
        return this.in_act;
    }
    
 
    @Override
    public void save(JSONObject obj) throws JSONException {
        obj.put("layer_type", this.layer_type);
        obj.put("out_sx", this.out_sx);
        obj.put("out_sy", this.out_sy);
        obj.put("out_depth", this.out_depth);
        obj.put("drop_prob", this.drop_prob);
    }

    @Override
    public void load(JSONObject obj) throws JSONException {
        this.layer_type = obj.getString("layer_type"); // defensive
        this.out_sx = obj.getInt("out_sx");
        this.out_sy = obj.getInt("out_sy");
        this.out_depth = obj.getInt("out_depth");
        this.drop_prob = obj.getDouble("drop_prob");
        this.dropped = new boolean[out_sx * out_sy * out_depth];
    }
}