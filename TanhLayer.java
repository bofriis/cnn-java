package rocks.propeller.karpathy;

import com.improaim.org.json.JSONException;
import com.improaim.org.json.JSONObject;

public class TanhLayer implements ILayer {

    int out_sx;
    int out_sy;
    int out_depth;
    String layer_type = "tanh";

    Vol in_act;
    Vol out_act;

    Util global = new Util();

    public TanhLayer(Definition opt) {
        this.out_sx = opt.in_sx;
        this.out_sy = opt.in_sy;
        this.out_depth = opt.in_depth;
    }

    @Override
    public Vol forward(Vol V, boolean is_training) {
        this.in_act = V;
        Vol V2 = V.cloneVol();
        int N = V.w.length;
        double[] V2w = V2.w;
        for(int i = 0; i < N; i++) {
            V2w[i] = Math.tanh(V2w[i]);
        }
        this.out_act = V2;
        return this.out_act;
    }

    @Override
    public void backward() {
        Vol V = this.in_act;
        Vol V2 = this.out_act;
        int N = V.w.length;
        V.dw = global.zeros(N);
        for(int i = 0; i < N; i++) {
            double t = V2.w[i];
            V.dw[i] = (1.0 - t * t) * V2.dw[i]; // (1 - tanh^2(x))
        }
    }

    @Override
    public ParamsAndGrads[] getParamsAndGrads() {
        return new ParamsAndGrads[0]; // Tanh har ingen parametre
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
    }

    @Override
    public void load(JSONObject json) throws JSONException {
        this.layer_type = json.getString("layer_type");
        this.out_sx = json.getInt("out_sx");
        this.out_sy = json.getInt("out_sy");
        this.out_depth = json.getInt("out_depth");
    }
}