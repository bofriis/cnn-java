package rocks.propeller.karpathy;

import com.improaim.org.json.JSONException;
import com.improaim.org.json.JSONObject;

/**
 * Implements a linear SVM layer (hinge loss), suitable for classification.
 */
public class SVMLayer implements ILayer {

    int out_sx;
    int out_sy;
    int out_depth;
    String layer_type = "svm";

    Vol in_act;
    Vol out_act;

    int y; // target label
    double margin = 1.0; // can be made configurable

    Util global = new Util();

    public SVMLayer(Definition opt) {
        this.out_sx = 1;
        this.out_sy = 1;
        this.out_depth = opt.out_depth;
        this.layer_type = "svm";
    }

    @Override
    public Vol forward(Vol V, boolean is_training) {
        this.in_act = V;
        this.out_act = V; // just return raw scores
        return V;
    }

    public double backward(int y) {
        // y is the correct label index
        this.y = y;

        Vol x = this.in_act;
        double[] xw = x.w;
        double[] xd = x.dw;

        // compute gradients and loss
        for (int i = 0; i < this.out_depth; i++) {
            if (i == y) {
                continue;
            }
            double marginLoss = -xw[y] + xw[i] + margin;
            if (marginLoss > 0) {
                xd[i] = 1.0;
                xd[y] -= 1.0;
            } else {
                xd[i] = 0.0;
            }
        }

        // loss is sum of positive hinge violations
        double loss = 0.0;
        for (int i = 0; i < this.out_depth; i++) {
            if (i == y) continue;
            double marginLoss = -xw[y] + xw[i] + margin;
            if (marginLoss > 0) {
                loss += marginLoss;
            }
        }
        return loss;
    }

    @Override
    public void backward() {
        // no-op: use `backward(int y)` instead with label
        throw new RuntimeException("Use backward(int y) with label");
    }

    @Override
    public ParamsAndGrads[] getParamsAndGrads() {
        return new ParamsAndGrads[0]; // no params in this layer
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
    public void save(JSONObject json) throws JSONException {
        json.put("layer_type", layer_type);
        json.put("out_sx", out_sx);
        json.put("out_sy", out_sy);
        json.put("out_depth", out_depth);
        json.put("margin", margin);
    }

    @Override
    public void load(JSONObject json) throws JSONException {
        this.layer_type = json.getString("layer_type");
        this.out_sx = json.getInt("out_sx");
        this.out_sy = json.getInt("out_sy");
        this.out_depth = json.getInt("out_depth");
        this.margin = json.has("margin") ? json.getDouble("margin") : 1.0;
    }
}