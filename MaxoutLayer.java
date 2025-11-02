package rocks.propeller.karpathy;

import com.improaim.org.json.JSONException;
import com.improaim.org.json.JSONObject;

public class MaxoutLayer implements ILayer {

    int out_sx;
    int out_sy;
    int out_depth;
    String layer_type = "maxout";

    Vol in_act;
    Vol out_act;

    int group_size; // antal lineære neuroner per maxout-gruppe
    int switches[]; // husker hvilken indeks der blev valgt under forward pass

    Util global = new Util();

    public MaxoutLayer(Definition opt) {
        this.group_size = opt.group_size > 0 ? opt.group_size : 2; // default 2

        // computed
        this.out_sx = opt.in_sx;
        this.out_sy = opt.in_sy;
        this.out_depth = (int) Math.floor(opt.in_depth / this.group_size);

        this.layer_type = "maxout";
    }

    @Override
    public Vol forward(Vol V, boolean is_training) {
        this.in_act = V;
        Vol A = new Vol(this.out_sx, this.out_sy, this.out_depth, 0.0);
        this.switches = new int[this.out_sx * this.out_sy * this.out_depth];

        for (int x = 0; x < this.out_sx; x++) {
            for (int y = 0; y < this.out_sy; y++) {
                for (int i = 0; i < this.out_depth; i++) {
                    int offset = i * this.group_size;
                    double maxval = V.get(x, y, offset);
                    int maxix = offset;
                    for (int j = 1; j < this.group_size; j++) {
                        double val = V.get(x, y, offset + j);
                        if (val > maxval) {
                            maxval = val;
                            maxix = offset + j;
                        }
                    }
                    A.set(x, y, i, maxval);
                    switches[(this.out_sx * y + x) * this.out_depth + i] = maxix;
                }
            }
        }

        this.out_act = A;
        return this.out_act;
    }

    @Override
    public void backward() {
        Vol V = this.in_act;
        Vol A = this.out_act;
        V.dw = global.zeros(V.w.length); // nulstil gradienter

        for (int x = 0; x < this.out_sx; x++) {
            for (int y = 0; y < this.out_sy; y++) {
                for (int i = 0; i < this.out_depth; i++) {
                    int ix = (this.out_sx * y + x) * this.out_depth + i;
                    int selected = this.switches[ix];
                    double chain_grad = A.get_grad(x, y, i);
                    V.dw[(V.sx * y + x) * V.depth + selected] += chain_grad;
                }
            }
        }
    }

    @Override
    public ParamsAndGrads[] getParamsAndGrads() {
        return new ParamsAndGrads[0]; // Ingen parametre i Maxout
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
        obj.put("group_size", this.group_size);
    }

    @Override
    public void load(JSONObject obj) throws JSONException {
        this.layer_type = obj.getString("layer_type");
        this.out_sx = obj.getInt("out_sx");
        this.out_sy = obj.getInt("out_sy");
        this.out_depth = obj.getInt("out_depth");
        this.group_size = obj.getInt("group_size");
    }
}