import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_stitching.Stitcher;

public class Main {
    public static void main(String[] args) {
        VideoCapture cap = new VideoCapture(0);  // Initialize the camera, 0 for the default camera, or specify the camera index

        // Check if the camera is opened successfully
        if (!cap.isOpened()) {
            System.err.println("Error: Could not open camera.");
            System.exit(-1);
        }

        // Define parameters for photo capture
        int photoWidth = 1920;  // Width of the captured photo
        int photoHeight = 1080; // Height of the captured photo

        MatVector capturedFrames = new MatVector();

        // Capture a series of frames for the photo
        for (int i = 0; i < 5; ++i) {  // Capture 5 frames (you can adjust this number)
            Mat frame = new Mat();
            cap.read(frame);

            // Resize the frame to the desired photo dimensions
            opencv_imgcodecs.resize(frame, frame, new Size(photoWidth, photoHeight));

            // Append the frame to the vector
            capturedFrames.push_back(frame);
        }

        // Release the camera
        cap.release();

        // Stitch the captured frames into a panorama
        Stitcher stitcher = Stitcher.create();
        Mat panorama = new Mat();
        int status = stitcher.stitch(capturedFrames, panorama);

        if (status == Stitcher.OK) {
            // Save the stitched panorama as a 360° photo
            opencv_imgcodecs.imwrite("360_photo.jpg", panorama);

            // Display the 360° photo (you can use a GUI library for a more advanced user interface)
            org.bytedeco.opencv.global.opencv_highgui.imshow("360° Photo", panorama);
            org.bytedeco.opencv.global.opencv_highgui.waitKey(0);
        } else {
            System.err.println("Error during stitching");
        }
    }
}
