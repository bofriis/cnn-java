package rocks.propeller.karpathy;

import com.improaim.org.json.JSONException;
import com.improaim.org.json.JSONObject;

public class SigmoidLayer implements ILayer {

    int out_sx;
    int out_sy;
    int out_depth;
    String layer_type = "sigmoid";

    Vol in_act;
    Vol out_act;

    public SigmoidLayer(Definition opt) {
        this.out_sx = opt.in_sx;
        this.out_sy = opt.in_sy;
        this.out_depth = opt.in_depth;
    }

    @Override
    public Vol forward(Vol V, boolean is_training) {
        this.in_act = V;
        Vol V2 = V.cloneVol();
        for (int i = 0; i < V2.w.length; i++) {
            V2.w[i] = 1.0 / (1.0 + Math.exp(-V2.w[i])); // sigmoid activation
        }
        this.out_act = V2;
        return this.out_act;
    }

    @Override
    public void backward() {
        Vol V = this.in_act;
        Vol V2 = this.out_act;

        for (int i = 0; i < V.w.length; i++) {
            double s = V2.w[i];
            V.dw[i] = V2.dw[i] * s * (1.0 - s);
        }
    }

    public double backward(int label) {
        double y = (double) label;
        Vol V = this.out_act;
        Vol X = this.in_act;

        if (V.w == null || V.w.length == 0) throw new IllegalStateException("Forward must be called before backward.");

        double loss = 0.0;
        for (int i = 0; i < V.w.length; i++) {
            double p = V.w[i];
            double dy = p - y;
            X.dw[i] = dy;
            loss += -y * Math.log(p + 1e-8) - (1.0 - y) * Math.log(1.0 - p + 1e-8);
        }
        return loss / V.w.length;
    }

    @Override
    public ParamsAndGrads[] getParamsAndGrads() {
        return new ParamsAndGrads[0]; // No weights
    }

    @Override public int get_out_sx() { return out_sx; }
    @Override public int get_out_sy() { return out_sy; }
    @Override public int get_out_depth() { return out_depth; }
    @Override public String get_layer_type() { return layer_type; }
    @Override public Vol get_out_act() { return out_act; }
    @Override public void set_in_act(Vol v) { this.in_act = v; }
    @Override public Vol get_in_act() { return this.in_act; }
    
    @Override
    public void save(JSONObject json) throws JSONException {
        json.put("layer_type", layer_type);
        json.put("out_sx", out_sx);
        json.put("out_sy", out_sy);
        json.put("out_depth", out_depth);
    }

    @Override
    public void load(JSONObject json) throws JSONException {
        this.layer_type = json.getString("layer_type");
        this.out_sx = json.getInt("out_sx");
        this.out_sy = json.getInt("out_sy");
        this.out_depth = json.getInt("out_depth");
    }
}