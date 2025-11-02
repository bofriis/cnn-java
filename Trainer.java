package rocks.propeller.karpathy;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import com.improaim.org.json.JSONArray;
import com.improaim.org.json.JSONException;
import com.improaim.org.json.JSONObject;


public class Trainer {

    private final Net net;
    private double learningRate = 0.01;
    private double momentum = 0.9;
    private double l2Decay = 0.0005;
    private double l1Decay = 0.0;
    private String lossFunction = "softmax"; // "softmax", "svm", "regression"

    private int batchSize = 1;
    private String method = "sgd"; // sgd/adam/adagrad/adadelta/windowgrad/nesterov
    private double ro = 0.95, eps = 1e-8, beta1 = 0.9, beta2 = 0.999;

    private int iteration = 0;
    private final ArrayList<double[]> gsum = new ArrayList<>();
    private final ArrayList<double[]> xsum = new ArrayList<>();
    private final Util global = new Util();

    public Trainer(Net net, String opt) throws JSONException {
        this.net = net;

        if (opt == null) opt = new JSONObject().toString();
        JSONObject options = new JSONObject(opt);

        this.learningRate = options.optDouble("learning_rate", this.learningRate);
        this.momentum = options.optDouble("momentum", this.momentum);
        this.l2Decay = options.optDouble("l2_decay", this.l2Decay);
        this.l1Decay = options.optDouble("l1_decay", this.l1Decay);
        this.batchSize = options.optInt("batch_size", this.batchSize);
        this.method = options.optString("method", this.method);
        this.ro = options.optDouble("ro", this.ro);
        this.eps = options.optDouble("eps", this.eps);
        this.beta1 = options.optDouble("beta1", this.beta1);
        this.beta2 = options.optDouble("beta2", this.beta2);

        ILayer last = net.getLastLayer();
        String type = last.get_layer_type();
        if (type.equals("softmax")) this.lossFunction = "softmax";
        else if (type.equals("svm")) this.lossFunction = "svm";
        else if (type.equals("regression") || type.equals("sigmoid")) this.lossFunction = "regression";
        else throw new IllegalArgumentException("Unsupported final layer type: " + type);
    }

    public void trainBatch(List<Vol> inputs, List<Integer> labels) {
        for (ILayer layer : net.layers) {
            for (ParamsAndGrads pg : layer.getParamsAndGrads()) {
                for (int i = 0; i < pg.grads.length; i++) {
                    pg.grads[i] = 0;
                }
            }
        }

        double totalLoss = 0.0;
        for (int i = 0; i < inputs.size(); i++) {
            Vol x = inputs.get(i);
            int y = labels.get(i);
            net.forward(x, true);
            totalLoss += computeLoss(y);
            for (int j = net.layers.length - 2; j >= 0; j--) {
                net.layers[j].backward();
            }
        }

        updateWeights(inputs.size());
        

    }

    private double computeLoss(int y) {
        ILayer last = net.layers[net.layers.length - 1];
        switch (lossFunction) {
            case "softmax":
                if (last instanceof SoftmaxLayer) return ((SoftmaxLayer) last).backward(y);
                break;
            case "svm":
                if (last instanceof SVMLayer) return ((SVMLayer) last).backward(y);
                break;
            case "regression":
                if (last instanceof RegressionLayer) return ((RegressionLayer) last).backward(y);
                break;
            case "sigmoid":
                if (last instanceof SigmoidLayer) return ((SigmoidLayer) last).backward(y);
                break;
        }
        throw new IllegalStateException("Unsupported or mismatched loss function:" + lossFunction);
    }

    private void updateWeights(int batchSize) {
        ParamsAndGrads[] pglist = net.getParamsAndGrads();

        if (gsum.isEmpty() && (!method.equals("sgd") || momentum > 0.0)) {
            for (ParamsAndGrads pg : pglist) {
                gsum.add(global.zeros(pg.params.length));
                if (method.equals("adam") || method.equals("adadelta")) {
                    xsum.add(global.zeros(pg.params.length));
                } else {
                    xsum.add(new double[0]);
                }
            }
        }

        for (int i = 0; i < pglist.length; i++) {
            ParamsAndGrads pg = pglist[i];
            double[] p = pg.params;
            double[] g = pg.grads;
            double l2_mul = pg.l2_decay_mul != 0.0 ? pg.l2_decay_mul : 1.0;
            double l1_mul = pg.l1_decay_mul != 0.0 ? pg.l1_decay_mul : 1.0;

            for (int j = 0; j < p.length; j++) {
                double l2grad = l2Decay * l2_mul * p[j];
                double l1grad = l1Decay * l1_mul * (p[j] > 0 ? 1 : -1);
                double gij = (g[j] / batchSize) + l1grad + l2grad;

                double[] gsumi = gsum.get(i);
                double[] xsumi = xsum.get(i);

                switch (method) {
                    case "adam":
                        gsumi[j] = beta1 * gsumi[j] + (1 - beta1) * gij;
                        xsumi[j] = beta2 * xsumi[j] + (1 - beta2) * gij * gij;
                        double bc1 = gsumi[j] / (1 - Math.pow(beta1, iteration + 1));
                        double bc2 = xsumi[j] / (1 - Math.pow(beta2, iteration + 1));
                        p[j] -= learningRate * bc1 / (Math.sqrt(bc2) + eps);
                        break;
                    case "adagrad":
                        gsumi[j] += gij * gij;
                        p[j] -= learningRate * gij / Math.sqrt(gsumi[j] + eps);
                        break;
                    case "adadelta":
                        gsumi[j] = ro * gsumi[j] + (1 - ro) * gij * gij;
                        double dx = -Math.sqrt((xsumi[j] + eps) / (gsumi[j] + eps)) * gij;
                        xsumi[j] = ro * xsumi[j] + (1 - ro) * dx * dx;
                        p[j] += dx;
                        break;
                    case "windowgrad":
                        gsumi[j] = ro * gsumi[j] + (1 - ro) * gij * gij;
                        p[j] -= learningRate * gij / Math.sqrt(gsumi[j] + eps);
                        break;
                    case "nesterov":
                        double dxN = gsumi[j];
                        gsumi[j] = momentum * gsumi[j] + learningRate * gij;
                        dxN = momentum * dxN - (1 + momentum) * gsumi[j];
                        p[j] += dxN;
                        break;
                    default:
                        if (momentum > 0.0) {
                            dx = momentum * gsumi[j] - learningRate * gij;
                            gsumi[j] = dx;
                            p[j] += dx;
                        } else {
                            p[j] -= learningRate * gij;
                        }
                        break;
                }

                g[j] = 0.0;
            }
        }

        iteration++;
    }

    public void saveState(File path) throws IOException, JSONException {
        JSONObject root = new JSONObject();
        root.put("iteration", iteration);
        root.put("gsum", serializeArrays(gsum));
        root.put("xsum", serializeArrays(xsum));

        try (FileWriter fw = new FileWriter(path)) {
            fw.write(root.toString(2));
        }
    }

    public void loadState(File path) throws IOException, JSONException {
        JSONObject root = new JSONObject(Files.readString(Paths.get(path.getAbsolutePath())));
        this.iteration = root.getInt("iteration");
        deserializeArrays(root.getJSONArray("gsum"), gsum);
        deserializeArrays(root.getJSONArray("xsum"), xsum);
    }

    private JSONArray serializeArrays(List<double[]> list) throws JSONException {
        JSONArray outer = new JSONArray();
        for (double[] arr : list) {
            JSONArray inner = new JSONArray();
            for (double v : arr) inner.put(v);
            outer.put(inner);
        }
        return outer;
    }

    private void deserializeArrays(JSONArray outer, List<double[]> list) throws JSONException {
        list.clear();
        for (int i = 0; i < outer.length(); i++) {
            JSONArray inner = outer.getJSONArray(i);
            double[] arr = new double[inner.length()];
            for (int j = 0; j < arr.length; j++) arr[j] = inner.getDouble(j);
            list.add(arr);
        }
    }

	public int getBatchSize() {
		return batchSize;
	}
}