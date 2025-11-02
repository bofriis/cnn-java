/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package rocks.propeller.karpathy;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;

import com.improaim.org.json.JSONArray;
import com.improaim.org.json.JSONException;
import com.improaim.org.json.JSONObject;

/**
 *
 * @author gola
 */
public class JsonUtil {
    
    void toJSON1(File filePath, ILayer[] layers) throws JSONException, IOException {
    
        JSONObject network = new JSONObject();
        JSONArray jsonLayers = new JSONArray();
        network.put("layers", jsonLayers);        
                
        for(ILayer layer : layers) {
            switch(layer.get_layer_type()) {
                case "conv": {
                    JSONObject jsonLayer = new JSONObject();
                    ConvLayer l = (ConvLayer)layer;
                    jsonLayer.put("sx", l.sx);
                    jsonLayer.put("sy", l.sy);
                    jsonLayer.put("stride", l.stride);
                    jsonLayer.put("in_depth", l.in_depth);
                    jsonLayer.put("out_depth", l.out_depth);
                    jsonLayer.put("out_sx", l.out_sx);
                    jsonLayer.put("out_sy", l.out_sy);
                    jsonLayer.put("layer_type", l.layer_type);
                    jsonLayer.put("l1_decay_mul", l.l1_decay_mul);
                    jsonLayer.put("l2_decay_mul", l.l2_decay_mul);
                    jsonLayer.put("pad", l.pad);
                    
                    JSONArray jsonFilters = new JSONArray();
                    for (Vol filter : l.filters) {
                        JSONObject jsonFilter = new JSONObject();
                        jsonFilter.put("sx", filter.sx);
                        jsonFilter.put("sy", filter.sy);
                        jsonFilter.put("depth", filter.depth);
                    
                        JSONObject jsonW = new JSONObject();
                        for(int wI=0; wI<filter.w.length; wI++) { 
                            jsonW.put((String.valueOf(wI)), filter.w[wI]);
                        }
                        jsonFilter.put("w", jsonW);
                        jsonFilters.put(jsonFilter);
                    }
                                        
                    JSONObject jsonBiases = new JSONObject();
                    jsonBiases.put("sx", l.biases.sx);
                    jsonBiases.put("sy", l.biases.sy);
                    jsonBiases.put("depth", l.biases.depth);
                    
                    JSONObject jsonW = new JSONObject();
                    for(int wI=0; wI<l.biases.w.length; wI++) { 
                        jsonW.put((String.valueOf(wI)), l.biases.w[wI]);
                    }
                    
                    jsonBiases.put("w", jsonW);
                    jsonLayer.put("biases", jsonBiases);
                    jsonLayers.put(jsonLayer);
                                       
                    break;
                }
                case "fc": {
                    JSONObject jsonLayer = new JSONObject();
                    FullyConnLayer l = (FullyConnLayer)layer;
                    jsonLayer.put("out_depth", l.out_depth);
                    jsonLayer.put("out_sx", l.out_sx);
                    jsonLayer.put("out_sy", l.out_sy);
                    jsonLayer.put("num_inputs", l.num_inputs);
                    jsonLayer.put("l1_decay_mul", l.l1_decay_mul);
                    jsonLayer.put("l2_decay_mul", l.l2_decay_mul);                    
                    jsonLayer.put("layer_type", l.layer_type);
                    
                    JSONArray jsonFilters = new JSONArray();
                    for (Vol filter : l.filters) {
                        JSONObject jsonFilter = new JSONObject();
                        jsonFilter.put("sx", filter.sx);
                        jsonFilter.put("sy", filter.sy);
                        jsonFilter.put("depth", filter.depth);
                    
                        JSONObject jsonW = new JSONObject();
                        for(int wI=0; wI<filter.w.length; wI++) { 
                            jsonW.put((String.valueOf(wI)), filter.w[wI]);
                        }
                        jsonFilter.put("w", jsonW);
                        jsonFilters.put(jsonFilter);
                    }
                                        
                    JSONObject jsonBiases = new JSONObject();
                    jsonBiases.put("sx", l.biases.sx);
                    jsonBiases.put("sy", l.biases.sy);
                    jsonBiases.put("depth", l.biases.depth);
                    
                    JSONObject jsonW = new JSONObject();
                    for(int wI=0; wI<l.biases.w.length; wI++) { 
                        jsonW.put((String.valueOf(wI)), l.biases.w[wI]);
                    }
                    
                    jsonBiases.put("w", jsonW);
                    jsonLayer.put("biases", jsonBiases);
                    jsonLayers.put(jsonLayer);
                                       
                    break;
                }
                case "input": { 
                    JSONObject jsonLayer = new JSONObject();
                    InputLayer l = (InputLayer)layer;
                    jsonLayer.put("out_depth", l.out_depth);
                    jsonLayer.put("out_sx", l.out_sx);
                    jsonLayer.put("out_sy", l.out_sy);
                    jsonLayer.put("layer_type", l.layer_type);
                    jsonLayers.put(jsonLayer);
                    break;
                }
                case "softmax": {
                    JSONObject jsonLayer = new JSONObject();
                    SoftmaxLayer l = (SoftmaxLayer)layer;
                    jsonLayer.put("out_depth", l.out_depth);
                    jsonLayer.put("out_sx", l.out_sx);
                    jsonLayer.put("out_sy", l.out_sy);
                    jsonLayer.put("num_inputs", l.num_inputs);
                    jsonLayer.put("layer_type", l.layer_type);

                    jsonLayers.put(jsonLayer);
                    break;
                }
                case "pool": {
                    JSONObject jsonLayer = new JSONObject();
                    PoolLayer l = (PoolLayer)layer;
                    jsonLayer.put("sx", l.sx);
                    jsonLayer.put("sy", l.sy);
                    jsonLayer.put("stride", l.stride);
                    jsonLayer.put("in_depth", l.in_depth);
                    jsonLayer.put("out_depth", l.out_depth);
                    jsonLayer.put("out_sx", l.out_sx);
                    jsonLayer.put("out_sy", l.out_sy);
                    jsonLayer.put("pad", l.pad);
                    jsonLayer.put("layer_type", l.layer_type);
                    jsonLayers.put(jsonLayer);             
                    break;
                }
                case "relu": {
                    JSONObject jsonLayer = new JSONObject();
                    ReluLayer l = (ReluLayer)layer;
                    jsonLayer.put("out_depth", l.out_depth);
                    jsonLayer.put("out_sx", l.out_sx);
                    jsonLayer.put("out_sy", l.out_sy);
                    jsonLayer.put("layer_type", l.layer_type);

                    jsonLayers.put(jsonLayer);           
                    break;
                }
                case "sigmoid": {
                    JSONObject jsonLayer = new JSONObject();
                    SigmoidLayer l = (SigmoidLayer)layer;
                    jsonLayer.put("out_depth", l.out_depth);
                    jsonLayer.put("out_sx", l.out_sx);
                    jsonLayer.put("out_sy", l.out_sy);
                    jsonLayer.put("layer_type", l.layer_type);
                    jsonLayers.put(jsonLayer);
                    break;
                }
                case "tanh": {
                    JSONObject jsonLayer = new JSONObject();
                    TanhLayer l = (TanhLayer)layer;
                    jsonLayer.put("out_depth", l.out_depth);
                    jsonLayer.put("out_sx", l.out_sx);
                    jsonLayer.put("out_sy", l.out_sy);
                    jsonLayer.put("layer_type", l.layer_type);
                    jsonLayers.put(jsonLayer);
                    break;
                }
                case "dropout": {
                    JSONObject jsonLayer = new JSONObject();
                    DropoutLayer l = (DropoutLayer) layer;
                    jsonLayer.put("out_depth", l.out_depth);
                    jsonLayer.put("out_sx", l.out_sx);
                    jsonLayer.put("out_sy", l.out_sy);
                    jsonLayer.put("drop_prob", l.drop_prob);
                    jsonLayer.put("layer_type", l.layer_type);
                    jsonLayers.put(jsonLayer);
                    break;
                }
                case "maxout": {
                    JSONObject jsonLayer = new JSONObject();
                    MaxoutLayer l = (MaxoutLayer) layer;
                    jsonLayer.put("out_depth", l.out_depth);
                    jsonLayer.put("out_sx", l.out_sx);
                    jsonLayer.put("out_sy", l.out_sy);
                    jsonLayer.put("group_size", l.group_size);
                    jsonLayer.put("layer_type", l.layer_type);
                    jsonLayers.put(jsonLayer);
                    break;
                }
                case "svm": {
                    JSONObject jsonLayer = new JSONObject();
                    SVMLayer l = (SVMLayer) layer;
                    jsonLayer.put("out_depth", l.out_depth);
                    jsonLayer.put("out_sx", l.out_sx);
                    jsonLayer.put("out_sy", l.out_sy);
                    jsonLayer.put("layer_type", l.layer_type);
                    jsonLayers.put(jsonLayer);
                    break;
                }
                case "regression": {
                    JSONObject jsonLayer = new JSONObject();
                    RegressionLayer l = (RegressionLayer) layer;
                    jsonLayer.put("out_depth", l.out_depth);
                    jsonLayer.put("out_sx", l.out_sx);
                    jsonLayer.put("out_sy", l.out_sy);
                    jsonLayer.put("layer_type", l.layer_type);
                    jsonLayers.put(jsonLayer);
                    break;
                }
            }
        }
        
        DataSet.crateTextFile(filePath, network.toString());
        
    }
    
    void fromJSON1(String filePath, ILayer[] layers) throws JSONException {
        
        ObjectMapper mapper = new ObjectMapper();
        JsonNode root = mapper.readTree(new java.io.File(filePath));
        JsonNode layersNode = root.path("layers");
        
        if (layersNode.size() != layers.length)
            throw new JSONException("Number of layers in JSON has to be equal to the number in the network");
        
        int layerI=0;
        for (Iterator<JsonNode> i = layersNode.elements(); i.hasNext();) {
            JsonNode layerNode = (JsonNode)i.next();
            ILayer layer = loadLayer(layerNode, layers[layerI]);
//            layers[layerI] = layer;
            layerI++;
        }        
    }
    
    public  class ObjectMapper {

		public JsonNode readTree(File file) {
			try(InputStream is = new FileInputStream(file)) {
				byte[] ba = is.readAllBytes();
				String s = new String(ba);
				JSONObject json = new JSONObject(s);
				JsonNode n = new JsonNode(json);
				return n;
			} catch(Exception e) {
				e.printStackTrace();
				return null;
			}
		}
    	
    }
    
    public  class JsonNode  {
    	JSONObject root;
    	JSONArray array;
    	Object namevalue;

    	JsonNode(Object o) {
    		if (o instanceof JSONObject) {
    			root=(JSONObject)o;
    		} else
    		if (o instanceof JSONArray) {
    			array=(JSONArray)o;
    		} else
    			namevalue=o;
    	}
    	public JsonNode path1(String key) throws JSONException {
//    		if (root!=null)
//    			return JsonNode(root.getJSONObject(key));
    		if (root.getJSONArray(key)!=null)
    			return new JsonNode(root.getJSONArray(key));
//    		return new JsonNode(root.getJSONObject(key)); //new JsonNode(root.getJSONArray(key));
    		if (root.getJSONObject(key)!=null)
    			return new JsonNode(root.getJSONObject(key));
    		return new JsonNode(root.get(key));
//    		return ne;
    	}
    	public JsonNode path(String key) throws JSONException {
    	    if (root.has(key)) {
    	        Object obj = root.get(key);
    	        if (obj instanceof JSONObject) return new JsonNode((JSONObject)obj);
    	        if (obj instanceof JSONArray) return new JsonNode((JSONArray)obj);
    	        return new JsonNode(obj);
    	    }
    	    return new JsonNode(JSONObject.NULL);
    	}

		public int asInt() {
//			return root.
			return (int)(namevalue);
		}

		public double asDouble() {
			// TODO Auto-generated method stub
			if (namevalue instanceof Integer)
				return (double)((Integer)namevalue).doubleValue();
			return (double)namevalue;
		}

		public String asText() {
			return (String)namevalue;
		}

	    public Iterator<JsonNode> elements() {
	    	if (array!=null)
	        return new Iterator<JsonNode>() {
	            int c = 0;

	            @Override
	            public boolean hasNext() {
	            	if (array!=null)
	                return c < array.length();
	            	
	            	return false;
	            			
	            }

	            @Override
	            public JsonNode next() {
	                try {
	                    if (array != null && c < array.length() && array.getJSONObject(c)!=null) {
	                        return new JsonNode(array.getJSONObject(c++));
	                    } else {
	                        throw new IndexOutOfBoundsException("No more elements");
	                    }
	                } catch (Exception e) {
	                    e.printStackTrace();
	                    return null;
	                }
	            }
	        };
	        
	        if (root != null) {
	            return new Iterator<JsonNode>() {
	                int c = 0;
	                int len = root.length();
	                List<JsonNode> res = new ArrayList<>();

	                {
	                    for (int i = 0; i < len; i++) {
	                    	try {
	                        res.add(new JsonNode(root.get(Integer.toString(i))));
	                    	} catch(Exception e) {
	                    		e.printStackTrace();
	                    	}
	                    }
	                }

	                @Override
	                public boolean hasNext() {
	                    return c < res.size();
	                }

	                @Override
	                public JsonNode next() {
	                    if (hasNext()) {
	                        return res.get(c++);
	                    } else {
	                        throw new NoSuchElementException("No more elements");
	                    }
	                }
	            };
	        }
	        return null;
	    }

		public int size() {
		
			return array.length();
		}
    	
    	
    }
    
    ILayer loadLayer(JsonNode layerNode, ILayer layer) throws JSONException {
        
        String layer_type = layerNode.path("layer_type").asText();
        
        if (!layer.get_layer_type().equals(layer_type))
            throw new JSONException("Layers are not of same type"); 
        
        switch(layer_type) {
                case "conv": {
                    ConvLayer l = (ConvLayer)layer;
                    l.out_depth = layerNode.path("out_depth").asInt();
                    l.out_sx = layerNode.path("out_sx").asInt();
                    l.out_sy = layerNode.path("out_sy").asInt();
                    l.sx = layerNode.path("sx").asInt();
                    l.sy = layerNode.path("sy").asInt();
                    l.stride = layerNode.path("stride").asInt();
                    l.in_depth = layerNode.path("in_depth").asInt();
                    l.l1_decay_mul = layerNode.path("l1_decay_mul").asDouble();
                    l.l2_decay_mul = layerNode.path("l2_decay_mul").asDouble();
                    l.pad = layerNode.path("pad").asInt();                                        

                    JsonNode biasesNode = layerNode.path("biases");
                    l.biases = new Vol(
                            biasesNode.path("sx").asInt(), 
                            biasesNode.path("sy").asInt(), 
                            biasesNode.path("depth").asInt(), 
                            0.0);

                    int wI=0;
                    for (Iterator<JsonNode>  i = biasesNode.path("w").elements(); i.hasNext();) {
                        JsonNode wNode = (JsonNode)i.next();
                        l.biases.w[wI++] = wNode.asDouble();
                    }
                    
//                    int wI = 0;
//                    JSONObject jsonW = biasesNode.path("w").root;
//                    for (String key : jsonW.keySet()) {
//                        l.biases.w[wI++] = jsonW.getDouble(key);
//                    }
//                    for (int i=0; i<biasesNode.path("w").size(); i++) {
//                        JsonNode wNode = biasesNode.path("w");
//                        l.biases.w[wI++] = wNode.asDouble();
//                    }
                    
                    int filI=0;
                    JsonNode filtersNode = layerNode.path("filters");
                    l.filters = new Vol[filtersNode.size()];
                    for (Iterator<JsonNode> i = filtersNode.elements(); i.hasNext();) {
                        JsonNode volNode = (JsonNode)i.next();
                        Vol v = new Vol(
                            volNode.path("sx").asInt(), 
                            volNode.path("sy").asInt(), 
                            volNode.path("depth").asInt(), 
                            0.0);
                        
                        int filwI=0;
                        for (Iterator<JsonNode>  j = volNode.path("w").elements(); j.hasNext();) {
                            JsonNode wNode = (JsonNode)j.next();
                            v.w[filwI++] = wNode.asDouble();
                        }
                        
                        l.filters[filI++] = v;
                    }
                    
                    break;
                }
                case "fc": {
                    FullyConnLayer l = (FullyConnLayer)layer;
                    l.out_depth = layerNode.path("out_depth").asInt();
                    l.out_sx = layerNode.path("out_sx").asInt();
                    l.out_sy = layerNode.path("out_sy").asInt();
                    l.num_inputs = layerNode.path("num_inputs").asInt();
                    l.l1_decay_mul = layerNode.path("l1_decay_mul").asDouble();
                    l.l2_decay_mul = layerNode.path("l2_decay_mul").asDouble();

                    JsonNode biasesNode = layerNode.path("biases");
                    l.biases = new Vol(
                            biasesNode.path("sx").asInt(), 
                            biasesNode.path("sy").asInt(), 
                            biasesNode.path("depth").asInt(), 
                            0.0);
                    
                    int wI=0;
                    for (Iterator i = biasesNode.path("w").elements(); i.hasNext();) {
                        JsonNode wNode = (JsonNode)i.next();
                        l.biases.w[wI++] = wNode.asDouble();
                    } 

                    int filI=0;
                    JsonNode filtersNode = layerNode.path("filters");
                    l.filters = new Vol[filtersNode.size()];
                    for (Iterator i = filtersNode.elements(); i.hasNext();) {
                        JsonNode volNode = (JsonNode)i.next();
                        Vol v = new Vol(
                            volNode.path("sx").asInt(), 
                            volNode.path("sy").asInt(), 
                            volNode.path("depth").asInt(), 
                            0.0);
                        
                        int filwI=0;
                        for (Iterator j = volNode.path("w").elements(); j.hasNext();) {
                            JsonNode wNode = (JsonNode)j.next();
                            v.w[filwI++] = wNode.asDouble();
                        }
                        
                        l.filters[filI++] = v;
                    }
                                
                    break;                    
                }
                case "input": {  
                    InputLayer l = (InputLayer)layer;
                    l.out_depth = layerNode.path("out_depth").asInt();
                    l.out_sx = layerNode.path("out_sx").asInt();
                    l.out_sy = layerNode.path("out_sy").asInt();
                                        
                    break;
                }
                case "softmax": {
                    SoftmaxLayer l = (SoftmaxLayer)layer;
                    l.out_depth = layerNode.path("out_depth").asInt();
                    l.out_sx = layerNode.path("out_sx").asInt();
                    l.out_sy = layerNode.path("out_sy").asInt();
                    l.num_inputs = layerNode.path("num_inputs").asInt();

                    break;
                }
                case "pool": {
                    PoolLayer l = (PoolLayer)layer;
                    l.out_depth = layerNode.path("out_depth").asInt();
                    l.out_sx = layerNode.path("out_sx").asInt();
                    l.out_sy = layerNode.path("out_sy").asInt();
                    l.sx = layerNode.path("sx").asInt();
                    l.sy = layerNode.path("sy").asInt();
                    l.stride = layerNode.path("stride").asInt();
                    l.in_depth = layerNode.path("in_depth").asInt();
                    l.pad = layerNode.path("pad").asInt();
                    //l.switchx = global.zeros(l.out_sx*l.out_sy*l.out_depth); // need to re-init these appropriately
                    //l.switchy = global.zeros(l.out_sx*l.out_sy*l.out_depth);
                    
                    break;
                }
                case "relu": {
                    ReluLayer l = (ReluLayer)layer;    
                    l.out_depth = layerNode.path("out_depth").asInt();
                    l.out_sx = layerNode.path("out_sx").asInt();
                    l.out_sy = layerNode.path("out_sy").asInt();
                    
                    break;
                }
                case "sigmoid": {
                    SigmoidLayer l = (SigmoidLayer)layer;    
                    l.out_depth = layerNode.path("out_depth").asInt();
                    l.out_sx = layerNode.path("out_sx").asInt();
                    l.out_sy = layerNode.path("out_sy").asInt();
                    
                    break;
                }
                case "dropout": {
                    DropoutLayer l = (DropoutLayer)layer;
                    l.out_depth = layerNode.path("out_depth").asInt();
                    l.out_sx = layerNode.path("out_sx").asInt();
                    l.out_sy = layerNode.path("out_sy").asInt();
                    l.drop_prob = layerNode.path("drop_prob").asDouble();
                    break;
                }
                case "tanh": {
                    TanhLayer l = (TanhLayer)layer;
                    l.out_depth = layerNode.path("out_depth").asInt();
                    l.out_sx = layerNode.path("out_sx").asInt();
                    l.out_sy = layerNode.path("out_sy").asInt();
                    break;
                }
                case "maxout": {
                    MaxoutLayer l = (MaxoutLayer)layer;
                    l.out_depth = layerNode.path("out_depth").asInt();
                    l.out_sx = layerNode.path("out_sx").asInt();
                    l.out_sy = layerNode.path("out_sy").asInt();
                    l.group_size = layerNode.path("group_size").asInt();
                    break;
                }
                case "svm": {
                    SVMLayer l = (SVMLayer)layer;
                    l.out_depth = layerNode.path("out_depth").asInt();
                    l.out_sx = layerNode.path("out_sx").asInt();
                    l.out_sy = layerNode.path("out_sy").asInt();
                    break;
                }
                case "regression": {
                    RegressionLayer l = (RegressionLayer) layer;
                    l.out_depth = layerNode.path("out_depth").asInt();
                    l.out_sx = layerNode.path("out_sx").asInt();
                    l.out_sy = layerNode.path("out_sy").asInt();
                    break;
                }
            }
        
        return layer;
    }
    
    public void toJSON(String filePath, ILayer[] layers) throws Exception {
        JSONObject network = new JSONObject();
        JSONArray jsonLayers = new JSONArray();
        network.put("layers", jsonLayers);

        for (ILayer layer : layers) {
            JSONObject jsonLayer = new JSONObject();
            jsonLayer.put("layer_type", layer.get_layer_type());

            switch (layer.get_layer_type()) {
                case "input": {
                    InputLayer l = (InputLayer) layer;
                    jsonLayer.put("out_depth", l.out_depth);
                    jsonLayer.put("out_sx", l.out_sx);
                    jsonLayer.put("out_sy", l.out_sy);
                    break;
                }
                case "conv": {
                    ConvLayer l = (ConvLayer) layer;
                    jsonLayer.put("sx", l.sx);
                    jsonLayer.put("sy", l.sy);
                    jsonLayer.put("stride", l.stride);
                    jsonLayer.put("in_depth", l.in_depth);
                    jsonLayer.put("out_depth", l.out_depth);
                    jsonLayer.put("out_sx", l.out_sx);
                    jsonLayer.put("out_sy", l.out_sy);
                    jsonLayer.put("l1_decay_mul", l.l1_decay_mul);
                    jsonLayer.put("l2_decay_mul", l.l2_decay_mul);
                    jsonLayer.put("pad", l.pad);

                    JSONArray filters = new JSONArray();
                    for (Vol v : l.filters) {
                        JSONArray w = new JSONArray();
                        for (double val : v.w) w.put(val);
                        filters.put(w);
                    }
                    jsonLayer.put("filters", filters);

                    JSONArray biases = new JSONArray();
                    for (double val : l.biases.w) biases.put(val);
                    jsonLayer.put("biases", biases);
                    break;
                }
                case "fc": {
                    FullyConnLayer l = (FullyConnLayer) layer;
                    jsonLayer.put("num_inputs", l.num_inputs);
                    jsonLayer.put("out_depth", l.out_depth);
                    jsonLayer.put("out_sx", l.out_sx);
                    jsonLayer.put("out_sy", l.out_sy);
                    jsonLayer.put("l1_decay_mul", l.l1_decay_mul);
                    jsonLayer.put("l2_decay_mul", l.l2_decay_mul);

                    JSONArray filters = new JSONArray();
                    for (Vol v : l.filters) {
                        JSONArray w = new JSONArray();
                        for (double val : v.w) w.put(val);
                        filters.put(w);
                    }
                    jsonLayer.put("filters", filters);

                    JSONArray biases = new JSONArray();
                    for (double val : l.biases.w) biases.put(val);
                    jsonLayer.put("biases", biases);
                    break;
                }
                case "relu":
                case "sigmoid":
                case "tanh":
                case "dropout":
                case "softmax":
                case "svm":
                case "regression":
                case "maxout":
                case "pool": {
                    layer.save(jsonLayer);
                    break;
                }
            }

            jsonLayers.put(jsonLayer);
        }

        try (FileWriter fw = new FileWriter(filePath)) {
            fw.write(network.toString(2));
        }
    }

    public void fromJSON(String filePath, ILayer[] layers) throws Exception {
        JSONObject root;
        try (InputStream is = new FileInputStream(new File(filePath))) {
            byte[] ba = is.readAllBytes();
            root = new JSONObject(new String(ba));
        }

        JSONArray layerArray = root.getJSONArray("layers");
        if (layerArray.length() != layers.length)
            throw new JSONException("Mismatch in layer count");

        for (int i = 0; i < layerArray.length(); i++) {
            JSONObject jsonLayer = layerArray.getJSONObject(i);
            String type = jsonLayer.getString("layer_type");
            ILayer layer = layers[i];
            if (!layer.get_layer_type().equals(type))
                throw new JSONException("Layer type mismatch at index " + i);

            switch (type) {
                case "conv": {
                    ConvLayer l = (ConvLayer) layer;
                    l.out_depth = jsonLayer.getInt("out_depth");
                    l.out_sx = jsonLayer.getInt("out_sx");
                    l.out_sy = jsonLayer.getInt("out_sy");
                    l.sx = jsonLayer.getInt("sx");
                    l.sy = jsonLayer.getInt("sy");
                    l.stride = jsonLayer.getInt("stride");
                    l.in_depth = jsonLayer.getInt("in_depth");
                    l.l1_decay_mul = jsonLayer.getDouble("l1_decay_mul");
                    l.l2_decay_mul = jsonLayer.getDouble("l2_decay_mul");
                    l.pad = jsonLayer.getInt("pad");

                    JSONArray filters = jsonLayer.getJSONArray("filters");
                    l.filters = new Vol[filters.length()];
                    for (int j = 0; j < filters.length(); j++) {
                        JSONArray w = filters.getJSONArray(j);
                        l.filters[j] = new Vol(1, 1, w.length(), 0.0);
                        for (int k = 0; k < w.length(); k++) l.filters[j].w[k] = w.getDouble(k);
                    }

                    JSONArray biases = jsonLayer.getJSONArray("biases");
                    l.biases = new Vol(1, 1, biases.length(), 0.0);
                    for (int j = 0; j < biases.length(); j++) l.biases.w[j] = biases.getDouble(j);
                    break;
                }
                case "fc": {
                    FullyConnLayer l = (FullyConnLayer) layer;
                    l.num_inputs = jsonLayer.getInt("num_inputs");
                    l.out_depth = jsonLayer.getInt("out_depth");
                    l.out_sx = jsonLayer.getInt("out_sx");
                    l.out_sy = jsonLayer.getInt("out_sy");
                    l.l1_decay_mul = jsonLayer.getDouble("l1_decay_mul");
                    l.l2_decay_mul = jsonLayer.getDouble("l2_decay_mul");

                    JSONArray filters = jsonLayer.getJSONArray("filters");
                    l.filters = new Vol[filters.length()];
                    for (int j = 0; j < filters.length(); j++) {
                        JSONArray w = filters.getJSONArray(j);
                        l.filters[j] = new Vol(1, 1, w.length(), 0.0);
                        for (int k = 0; k < w.length(); k++) l.filters[j].w[k] = w.getDouble(k);
                    }

                    JSONArray biases = jsonLayer.getJSONArray("biases");
                    l.biases = new Vol(1, 1, biases.length(), 0.0);
                    for (int j = 0; j < biases.length(); j++) l.biases.w[j] = biases.getDouble(j);
                    break;
                }
                case "input": {
                    InputLayer l = (InputLayer) layer;
                    l.out_depth = jsonLayer.getInt("out_depth");
                    l.out_sx = jsonLayer.getInt("out_sx");
                    l.out_sy = jsonLayer.getInt("out_sy");
                    break;
                }
                case "relu":
                case "sigmoid":
                case "tanh":
                case "dropout":
                case "softmax":
                case "svm":
                case "regression":
                case "maxout":
                case "pool": {
                    layer.load(jsonLayer);
                    break;
                }
            }
        }
    }
}
