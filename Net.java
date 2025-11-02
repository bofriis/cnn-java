/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package rocks.propeller.karpathy;

import java.io.BufferedInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import com.improaim.org.json.JSONArray;
import com.improaim.org.json.JSONException;
import com.improaim.org.json.JSONObject;
import com.improaim.shared.math.Matrix32;

/**
 *
 * @author gola
 */
public class Net {

	ILayer[] layers;

	Util global = new Util();
	JsonUtil jutil = new JsonUtil();

	private double learningRate = 0.01;
	private double momentum = 0.0;
	private double l2Decay = 0.0;
	private double l1Decay = 0.0;

	public void setLearningRate(double lr) {
		this.learningRate = lr;
	}

	public void setMomentum(double m) {
		this.momentum = m;
	}

	public void setL2Decay(double decay) {
		this.l2Decay = decay;
	}

	public void setL1Decay(double decay) {
		this.l1Decay = decay;
	}

	public ILayer getLayer(int i) {
		return layers[i];
	}

	public int layers() {
		return layers.length;
	}

	// takes a list of layer definitions and creates the network layer objects
	public void makeLayers(String[] layer_defs) throws JSONException {

		// few checks
		// global.Assert(defs.length >= 2, "Error! At least one input layer and one loss
		// layer are required.") throws Exception;
		// global.Assert(defs[0].getString("type") == "input", "Error! First layer must
		// be the input layer, to declare size of inputs");

		Definition[] defs = desugar(layer_defs);

		// create the layers
		layers = new ILayer[defs.length];
		for (int i = 0; i < defs.length; i++) {
			Definition def = defs[i];
			if (i > 0) {
				ILayer prev = layers[i - 1];
				def.in_sx = (int) prev.get_out_sx();
				def.in_sy = (int) prev.get_out_sy();
				def.in_depth = prev.get_out_depth();
			}

			switch (def.type) {
			case "fc":
				layers[i] = new FullyConnLayer(def);
				break;
			// case "lrn": layers[i] = new LocalResponseNormalizationLayer(def); break;
			case "dropout":
				layers[i] = new DropoutLayer(def);
				break;
			case "input":
				layers[i] = new InputLayer(def);
				break;
			case "softmax":
				layers[i] = new SoftmaxLayer(def);
				break;
			// case "regression": layers[i] = new RegressionLayer(def); break;
			case "conv":
				layers[i] = new ConvLayer(def);
				break;
			case "pool":
				layers[i] = new PoolLayer(def);
				break;
			case "relu":
				layers[i] = new ReluLayer(def);
				break;
			case "sigmoid":
				layers[i] = new SigmoidLayer(def);
				break;
			case "tanh":
				layers[i] = new TanhLayer(def);
				break;
			case "maxout":
				layers[i] = new MaxoutLayer(def);
				break;
			case "svm":
				layers[i] = new SVMLayer(def);
				break;
			default:
				System.out.println("ERROR: UNRECOGNIZED LAYER TYPE: " + def.type);
			}
		}

	}

	// desugar layer_defs for adding activation, dropout layers etc
	Definition[] desugar(String[] layer_defs) throws JSONException {

		JSONObject[] defs = new JSONObject[layer_defs.length];
		for (int i = 0; i < layer_defs.length; i++) {
			defs[i] = new JSONObject(layer_defs[i]);
		}

		ArrayList<Definition> new_defs = new ArrayList<Definition>();
		for (int i = 0; i < defs.length; i++) {
			JSONObject def = defs[i];

			if (def.getString("type").equals("softmax") || def.getString("type").equals("svm")) {
				// add an fc layer here, there is no reason the user should
				// have to worry about this and we almost always want to
				new_defs.add(new Definition("fc", def.getInt("num_classes")));
			}

			if (def.getString("type").equals("regression")) {
				// add an fc layer here, there is no reason the user should
				// have to worry about this and we almost always want to
				new_defs.add(new Definition("fc", def.getInt("num_neurons")));
			}

			Definition layer = new Definition(def.getString("type"));

			if ((def.getString("type").equals("fc") || def.getString("type").equals("conv")) && def.has("bias_pref")) {
				layer.bias_pref = 0.0;
				if (def.has("activation") && def.getString("activation").equals("relu")) {
					layer.bias_pref = 0.1; // relus like a bit of positive bias to get gradients early
					// otherwise it's technically possible that a relu unit will never turn on (by
					// chance)
					// and will never get any gradient and never contribute any computation. Dead
					// relu.
				}
			}

			if (def.has("out_sx"))
				layer.out_sx = def.getInt("out_sx");
			if (def.has("out_sy"))
				layer.out_sy = def.getInt("out_sy");
			if (def.has("out_depth"))
				layer.out_depth = def.getInt("out_depth");

			if (def.has("sx"))
				layer.sx = def.getInt("sx");
			if (def.has("filters"))
				layer.filters = def.getInt("filters");
			if (def.has("stride"))
				layer.stride = def.getInt("stride");
			if (def.has("pad"))
				layer.pad = def.getInt("pad");
			if (def.has("activation"))
				layer.activation = def.getString("activation");
			if (def.has("num_classes"))
				layer.num_neurons = def.getInt("num_classes");
			if (def.has("num_neurons"))
				layer.num_neurons = def.getInt("num_neurons");
			new_defs.add(layer);

			if (def.has("activation")) {
				if (def.getString("activation").equals("relu")) {
					new_defs.add(new Definition("relu"));
				} else if (def.getString("activation").equals("sigmoid")) {
					new_defs.add(new Definition("sigmoid"));
				} else if (def.getString("activation").equals("tanh")) {
					new_defs.add(new Definition("tanh"));
				} else if (def.getString("activation").equals("maxout")) {
					// create maxout activation, and pass along group size, if provided
					int gs = def.has("group_size") ? def.getInt("group_size") : 2;
					new_defs.add(new Definition(gs, "maxout"));
				} else {
					// console.log('ERROR unsupported activation ' + def.activation);
					System.out.println("ERROR unsupported activation " + def.getString("activation"));
				}
			}

			if (def.has("drop_prob") && def.getString("type").equals("dropout")) {
				Definition temp = new Definition("dropout");
				temp.drop_prob = def.getDouble("drop_prob");
				new_defs.add(temp);
			}

		}
		return new_defs.toArray(new Definition[0]);
	}

	// forward prop the network.
	// The trainer class passes is_training = true, but when this function is
	// called from outside (not from the trainer), it defaults to prediction mode
	public Vol forward(Vol x, boolean isTraining) {
		Vol activation = x;
		for (int i = 0; i < layers.length; i++) {
			ILayer layer = layers[i];
			activation = layer.forward(activation, isTraining);
//	        System.out.println("Layer " + i + " (" + layer.get_layer_type() + ") output shape: "
//	            + activation.sx + "x" + activation.sy + "x" + activation.depth + " -> w.length=" + activation.w.length);
		}
		return activation;
	}

	double getCostLoss(Vol V, int y) {
		this.forward(V, false);
		int N = this.layers.length;
		double loss = ((SoftmaxLayer) this.layers[N - 1]).backward(y);
		return loss;
	}

	// backprop: compute gradients wrt all parameters
	double backward(int y) {
		int N = this.layers.length;
		ILayer last = this.layers[N - 1];
		double loss;

		if (last instanceof SoftmaxLayer) {
			loss = ((SoftmaxLayer) last).backward(y);
		} else if (last instanceof SVMLayer) {
			loss = ((SVMLayer) last).backward(y);
		} else if (last instanceof RegressionLayer) {
			loss = ((RegressionLayer) last).backward(y);
		} else if (last instanceof SigmoidLayer) {
			loss = ((SigmoidLayer) last).backward(y);
		} else {
			throw new IllegalStateException("Unknown loss layer type: " + last.getClass().getSimpleName());
		}

		for (int i = N - 2; i >= 0; i--) {
			this.layers[i].backward();
		}

		return loss;
	}

	ParamsAndGrads[] getParamsAndGrads() {
		// accumulate parameters and gradients for the entire network
		ArrayList<ParamsAndGrads> response = new ArrayList<ParamsAndGrads>();
		for (int i = 0; i < this.layers.length; i++) {
			ParamsAndGrads[] layer_reponse = this.layers[i].getParamsAndGrads();
			for (int j = 0; j < layer_reponse.length; j++) {
				response.add(layer_reponse[j]);
			}
		}
		return response.toArray(new ParamsAndGrads[0]);
	}

	public double getPrediction() {
		// this is a convenience function for returning the argmax
		// prediction, assuming the last layer of the net is a softmax
		ILayer S = this.layers[this.layers.length - 1];
		global.Assert(S.get_layer_type().equals("softmax"),
				"getPrediction function assumes softmax as last layer of the net!");

		double[] p = S.get_out_act().w;
		for (int i = 0; i < p.length; i++)
			System.out.println(String.format("%d=%.2f%%", i, p[i] * 100));
		double maxv = p[0];
		int maxi = 0;
		for (int i = 1; i < p.length; i++) {
			if (p[i] > maxv) {
				maxv = p[i];
				maxi = i;
			}
		}
		return maxi; // return index of the class with highest class probability
	}

	public void trainBatch(List<Vol> inputs, List<Integer> labels) {
		for (ILayer layer : layers) {
			for (ParamsAndGrads pg : layer.getParamsAndGrads()) {
				for (int i = 0; i < pg.grads.length; i++) {
					pg.grads[i] = 0.0;
				}
			}
		}
		for (int i = 0; i < inputs.size(); i++) {
			Vol x = inputs.get(i);
			int y = labels.get(i);
			forward(x, true);
			((SoftmaxLayer) layers[layers.length - 1]).backward(y);
			for (int j = layers.length - 2; j >= 0; j--) {
				layers[j].backward();
			}
		}
		updateWeights(inputs.size());
	}

	private void updateWeights(int batchSize) {
		for (ILayer layer : layers) {
			for (ParamsAndGrads pg : layer.getParamsAndGrads()) {
				for (int i = 0; i < pg.params.length; i++) {
					double l2grad = l2Decay * pg.params[i];
					double l1grad = l1Decay * (pg.params[i] > 0 ? 1 : -1);
					double grad = (pg.grads[i] / batchSize) + l2grad + l1grad;

					// momentum update
					pg.gsum[i] = momentum * pg.gsum[i] - learningRate * grad;
					pg.params[i] += pg.gsum[i];
				}
			}
		}
	}

	public void toJSON(File filePath) throws JSONException, IOException {
		JSONArray jsonLayers = new JSONArray();
		for (ILayer layer : layers) {
			JSONObject jsonLayer = new JSONObject();
			layer.save(jsonLayer);
			jsonLayers.put(jsonLayer);
		}
		JSONObject net = new JSONObject();
		net.put("layers", jsonLayers);
		DataSet.crateTextFile(filePath, net.toString(2));
		System.out.println("traning data saved:" + filePath.getAbsolutePath());
	}

	public void fromJSON(File filePath) throws JSONException, IOException {
		System.out.println("loading network and weights " + filePath.getAbsolutePath());
		String jsonContent = new String(Files.readAllBytes(Paths.get(filePath.getAbsolutePath())));
		loadNet(jsonContent);
	}

	private void loadNet(String jsonContent) throws JSONException {
		JSONObject net = new JSONObject(jsonContent);
		JSONArray jsonLayers = net.getJSONArray("layers");
		layers = new ILayer[jsonLayers.length()];
		for (int i = 0; i < jsonLayers.length(); i++) {
			JSONObject layerJson = jsonLayers.getJSONObject(i);
			String type = layerJson.getString("layer_type");
			switch (type) {
			case "input":
				layers[i] = new InputLayer();
				break;
			case "conv":
				layers[i] = new ConvLayer(new Definition("conv"));
				break;
			case "fc":
				layers[i] = new FullyConnLayer(new Definition("fc"));
				break;
			case "dropout":
				layers[i] = new DropoutLayer(new Definition("dropout"));
				break;
			case "softmax":
				layers[i] = new SoftmaxLayer(new Definition("softmax"));
				break;
			case "pool":
				layers[i] = new PoolLayer(new Definition("pool"));
				break;
			case "relu":
				layers[i] = new ReluLayer(new Definition("relu"));
				break;
			case "sigmoid":
				layers[i] = new SigmoidLayer(new Definition("sigmoid"));
				break;
			case "tanh":
				layers[i] = new TanhLayer(new Definition("tanh"));
				break;
			case "maxout":
				layers[i] = new MaxoutLayer(new Definition("maxout"));
				break;
			case "svm":
				layers[i] = new SVMLayer(new Definition("svm"));
				break;
			default:
				throw new JSONException("Unknown layer type: " + type);
			}
			layers[i].load(layerJson);
		}
	}

//	public void toJSON(String filePath) throws JSONException, IOException {
//		jutil.toJSON(filePath, layers);
//	}
//
//	public void fromJSON(String filePath) throws JSONException {
//		jutil.fromJSON(filePath, layers);
//	}

	public ILayer getLastLayer() {
		if (layers == null || layers.length == 0) {
			throw new IllegalStateException("Network has no layers defined.");
		}
		return layers[layers.length - 1];
	}

	public List<Matrix32> getConvWeightImages() {
		List<Matrix32> images = new ArrayList<>();

		for (ILayer layer : layers) {
			if (layer instanceof ConvLayer) {
				ConvLayer conv = (ConvLayer) layer;

				for (Vol filter : conv.filters) {
					int sx = filter.sx;
					int sy = filter.sy;
					int depth = filter.depth;

					// For visualisering kan vi vælge én kanal (f.eks. depth=0) eller snitte dem
					double[] w = filter.w;

					// Vi visualiserer kun første kanal (hvis flere)
					int size = sx * sy;
					if (w.length < size)
						continue; // sikkerhedstjek

					double min = Double.POSITIVE_INFINITY;
					double max = Double.NEGATIVE_INFINITY;
					for (int i = 0; i < size; i++) {
						double v = w[i];
						min = Math.min(min, v);
						max = Math.max(max, v);
					}

					Matrix32 img = new Matrix32(sx, sy, 1);
					for (int y = 0; y < sy; y++) {
						for (int x = 0; x < sx; x++) {
							int i = y * sx + x;
							double val = w[i];
							double norm = (val - min) / (max - min + 1e-8);
							img.set(x, y, norm); // normaliseret [0,1]
						}
					}

					images.add(img);
				}
			}
		}

		return images;
	}

	public List<Matrix32> getConvVolImages() {
		List<Matrix32> images = new ArrayList<>();

		for (ILayer layer : layers) {
			if (layer instanceof ConvLayer) {
				ConvLayer conv = (ConvLayer) layer;
				Vol out = conv.out_act;
				int depth = out.depth;

				for (int d = 0; d < depth; d++) {
					Matrix32 mat = out.toMatrix(d, 0); // d = channel index, 0 = batch index
					images.add(mat);
				}
			}
		}
		return images;

	}

	public List<Matrix32> getFCWeightImages(int inputWidth, int inputHeight, int maxNeurons) {
		List<Matrix32> images = new ArrayList<>();

		for (ILayer layer : layers) {
			if (layer instanceof FullyConnLayer) {
				FullyConnLayer fc = (FullyConnLayer) layer;

				int sx = inputWidth;
				int sy = inputHeight;
				int sz = fc.num_inputs / (sx * sy);

				for (int i = 0; i < Math.min(fc.out_depth, maxNeurons); i++) {
					double[] weights = fc.filters[i].w;

					// Hvis input ikke matcher forventet billedeform, spring over
					if (weights.length != sx * sy * sz)
						continue;

					Matrix32 img = new Matrix32(sx, sy, 1);

					// Vi tager kun første kanal (depth 0)
					double min = Double.POSITIVE_INFINITY;
					double max = Double.NEGATIVE_INFINITY;
					for (int j = 0; j < sx * sy; j++) {
						double v = weights[j];
						min = Math.min(min, v);
						max = Math.max(max, v);
					}

					for (int y = 0; y < sy; y++) {
						for (int x = 0; x < sx; x++) {
							int idx = y * sx + x;
							double norm = (weights[idx] - min) / (max - min + 1e-8);
							img.set(x, y, norm);
						}
					}

					images.add(img);
				}
			}
		}

		return images;
	}

	public void fromJSON(InputStream is) throws IOException, JSONException {
		StringBuffer sb = new StringBuffer();
		BufferedInputStream bais = new BufferedInputStream(is);
		while(bais.available()>0) {
			sb.append(new String(bais.readAllBytes()));
		}
		loadNet(sb.toString());
	}


}
