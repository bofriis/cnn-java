/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package rocks.propeller.karpathy;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;

import com.improaim.shared.math.ImageUtils;
import com.improaim.shared.math.Matrix32;

import javafx.scene.image.Image;
import javafx.scene.paint.Color;

/**
 *
 * @author gola
 */
public class VolUtil {

	Util global = new Util();

	// Volume utilities
	// intended for use with data augmentation
	// crop is the size of output
	// dx,dy are offset wrt incoming volume, of the shift
	// fliplr is boolean on whether we also want to flip left<->right
	Vol augment(Vol V, int crop, Integer dx, Integer dy, Boolean fliplr) {
		// note assumes square outputs of size crop x crop
		if (fliplr == null)
			fliplr = false;
		if (dx == null)
			dx = global.randi(0, V.sx - crop);
		if (dy == null)
			dy = global.randi(0, V.sy - crop);

		// randomly sample a crop in the input volume
		Vol W;
		if (crop != V.sx || dx != 0 || dy != 0) {
			W = new Vol(crop, crop, V.depth, 0.0);
			for (int x = 0; x < crop; x++) {
				for (int y = 0; y < crop; y++) {
					if (x + dx < 0 || x + dx >= V.sx || y + dy < 0 || y + dy >= V.sy)
						continue; // oob
					for (int d = 0; d < V.depth; d++) {
						W.set(x, y, d, V.get(x + dx, y + dy, d)); // copy data over
					}
				}
			}
		} else {
			W = V;
		}

		if (fliplr) {
			// flip volume horziontally
			Vol W2 = W.cloneAndZero();
			for (int x = 0; x < W.sx; x++) {
				for (int y = 0; y < W.sy; y++) {
					for (int d = 0; d < W.depth; d++) {
						W2.set(x, y, d, W.get(W.sx - x - 1, y, d)); // copy data over
					}
				}
			}
			W = W2; // swap
		}
		return W;
	}

	public Vol imagePath_to_vol(Matrix32 img, boolean convert_grayscale) {

		try {
			Image image = ImageUtils.matrixToImage(img);
			return img_to_vol(image, convert_grayscale);
		} catch (Exception ex) {
			System.out.println(ex);
		}
		return null;

	}

//	public Vol matrix_to_vol(Matrix32 m, boolean convert_grayscale) {
//
////		Image image = ImageUtils.matrixToImage(m);
//
//		return img_to_vol(image, convert_grayscale);
//
//	}

	public Vol matrix_to_vol(String file, boolean convert_grayscale) {

		Image image = null;
		try (InputStream is = new FileInputStream(new File(file))) {
			image = new Image(is);
			return img_to_vol(image, convert_grayscale);
		} catch (Exception ex) {
			System.out.println(ex);
		}
		return null;

	}
	
	public Vol matrix_to_vol(Matrix32 m) {
		int RGBA = 4, RGB = 3, BW = 1;
		int c = BW;
		float[] p = m.getRowPackedCopy();
		p = Matrix32.scale(p, -0.5f, 0.5f);
		Vol x = new Vol(m.cols(), m.rows(), c, 0.0); // input volume (image)
		x.w = Matrix32.toDouble(p);
		return x;
	}

	// img is a DOM element that contains a loaded image
	// returns a Vol of size (W, H, 3). 3 is for RGB
	Vol img_to_vol(Image img, boolean convert_grayscale) {

		int RGBA = 4, RGB = 3, BW = 1;

		// prepare the input: get pixels and normalize them
		int[] p = getRgbByteArray(img); // getByteArray(img);
		int W = (int) img.getWidth();
		int H = (int) img.getHeight();
		double[] pv = new double[p.length];
		for (int i = 0; i < p.length; i++) {
			pv[i] = p[i] / 255.0 - 0.5; // normalize image pixels to [-0.5, 0.5]
		}
		Vol x = new Vol(W, H, RGB, 0.0); // input volume (image)
		x.w = pv;

//		if (convert_grayscale) {
//			// flatten into depth=1 array
//			Vol x1 = new Vol(W, H, BW, 0.0);
//			for (int i = 0; i < W; i++) {
//				for (int j = 0; j < H; j++) {
//					x1.set(i, j, 0, x.get(i, j, 0));
//				}
//			}
//			x = x1;
//		}
		if (convert_grayscale) {
		    Vol x1 = new Vol(W, H, BW, 0.0);
		    for (int i = 0; i < W; i++) {
		        for (int j = 0; j < H; j++) {
		            double r = x.get(i, j, 0) + 0.5;
		            double g = x.get(i, j, 1) + 0.5;
		            double b = x.get(i, j, 2) + 0.5;
		            double gray = (0.299*r + 0.587*g + 0.114*b) - 0.5;
		            x1.set(i, j, 0, gray);
		        }
		    }
		    x = x1;
		}
		return x;
	}

	static int getColor(Image img, int x, int y) {
		Color color = img.getPixelReader().getColor(x, y);
//	int c = new AColor((float)color.getRed(), (float)color.getGreen(), (float)color.getBlue()).getRGB();
		int c = img.getPixelReader().getArgb(x, y);
		return c;
	}

	static int[] getByteArray(Image img) {

		int w = (int) img.getWidth();
		int h = (int) img.getHeight();

		int[] rgbs = new int[w * h];
		int off = 0;
//    img.getRGB(0, 0, w, h, rgbs, 0, w);

		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				rgbs[off++] = getColor(img, x, y);
			}
		}
		return rgbs;
	}

	static int[] getRgbByteArray(Image image) {

		int w = (int) image.getWidth();
		int h = (int) image.getHeight();
		int BLOCK_SIZE = 3;

		int[] pixels = getByteArray(image); // (new int[w * h];
//        image.getRGB(0, 0, w, h, pixels, 0, w);

		int[] rgbBytes = new int[w * h * BLOCK_SIZE];

		for (int r = 0; r < h; r++) {
			for (int c = 0; c < w; c++) {
				int index = r * w + c;
				int indexRgb = r * w * BLOCK_SIZE + c * BLOCK_SIZE;

//				rgbBytes[indexRgb] = (byte) ((pixels[index] >> 16) & 0xff);
//				rgbBytes[indexRgb + 1] = (byte) ((pixels[index] >> 8) & 0xff);
//				rgbBytes[indexRgb + 2] = (byte) (pixels[index] & 0xff);
				
				rgbBytes[indexRgb]     = (pixels[index] >> 16) & 0xff;
				rgbBytes[indexRgb + 1] = (pixels[index] >> 8) & 0xff;
				rgbBytes[indexRgb + 2] = pixels[index] & 0xff;
			}
		}

		return rgbBytes;
	}
}
