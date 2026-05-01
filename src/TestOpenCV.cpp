#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>

using namespace std;
using namespace cv;


int main() {

    cout << "OpenCV version: " << CV_VERSION << endl;

    // Open default camera
    VideoCapture cap(0, cv::CAP_V4L2); // Gebruik V4L2 backend (Linux) - pas aan indien nodig voor andere platforms
 
    // Check of camera kan worden geopend ...
    if (!cap.isOpened()) {
        cerr << "Can't open camera" << endl;
        return -1;
    }

    Mat src, hsv, mask1, mask2, mask3, mask;

    // Open de seriële poort (vervang "/dev/ttyUSB0" door de juiste poortnaam)
    int serial_port = open("/dev/ttyUSB0", O_RDWR);

    // Check of de seriële poort succesvol is geopend
    if (serial_port < 0) {
        perror("Error opening serial port");
        return -1;
    }

    termios tty;
    tcgetattr(serial_port, &tty);

    cfsetispeed(&tty, B115200);
    cfsetospeed(&tty, B115200);

    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;
    tty.c_cflag &= ~PARENB;
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CRTSCTS;

    tcsetattr(serial_port, TCSANOW, &tty);

    while (true) {
        
        // 1. Frame van camera lezen
        cap.read(src);


        // Variabele voor de afstand tussen de bal en de border
        double ActualPosition = 0; // Initialize the actual position

        ActualPosition = 10 + (10 * sin(10 * (double) getTickCount() / getTickFrequency())); // Simuleer een variërende positie (vervang dit met echte berekening)

        // Seriele communicatie met Arduino (vervang "COM3" door de juiste poortnaam)
        // SerialPort serial("COM3", 9600); // Baudrate moet overeenkomen

        string msg = "$" + to_string(ActualPosition) + "*\n";
        write(serial_port, msg.c_str(), msg.length());  

        //cout << "$" << ActualPosition << "*" << endl;

        // Display the image
        imshow("Source", src);


        // ESC = stoppen
        if (waitKey(1) == 27) break;
        
    }

    // Camera vrijgeven
    cap.release();

    // Alle vensters sluiten
    destroyAllWindows();

    return 0;
}