// +----------------------------------------------------------------------------------+
// | Project:      Knikkerbaan Regelsysteem                                           |
// | Bestand:      VisionKnikkerbaan.cpp                                              |
// | Auteur:       Florian van Doorn                                                  |
// | Opleiding:    Mechatronica - Avans Hogeschool Breda                              |
// | Datum:        28-05-2026                                                         |
// | Versie:       0.3                                                                |
// |                                                                                  |
// | Beschrijving:                                                                    |
// | Verwerkt camerabeelden met OpenCV om de positie van de knikker                   |
// | te bepalen voor de regelkring.                                                   |
// |                                                                                  |
// | Functionaliteit:                                                                 |
// | - Camera uitlezen                                                                |
// | - HSV filtering                                                                  |
// | - Objectdetectie                                                                 |
// | - Positiebepaling                                                                |
// |                                                                                  |
// | Libraries:                                                                       |
// | - OpenCV                                                                         |
// | - iostream                                                                       |
// | - algorithm                                                                      |
// | - fcntl.h                                                                        |
// | - unistd.h                                                                       |
// | - termios.h                                                                      |
// +----------------------------------------------------------------------------------+

// used libraries
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <string>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <GLFW/glfw3.h>
#include <GL/gl.h>
#include <cstdint>

// namespaces
using namespace std;
using namespace cv;

// Schermstaten voor navigatie
enum ScreenState
{
    SCREEN_HOME,
    SCREEN_THRESHOLD,
    SCREEN_COMM
};

// global variables
ScreenState currentScreen = SCREEN_HOME;

// Images and masks
Mat src;   // Input frame van camera
Mat hsv;   // HSV versie van het frame
Mat mask1; // Mask voor lage rood range
Mat mask2; // Mask voor hoge rood range
Mat mask3; // Mask voor blauw range
Mat mask;  // Gecombineerde mask voor beide rood ranges

// OpenGL textures for displaying images in ImGui
static GLuint texMask = 0;
static GLuint texMaskBlue = 0;
static GLuint texOverlay = 0;

// UI state
static int viewMode = 0; // 0 = rood mask, 1 = blauw mask, 2 = overlay
static bool autoscaleImages = true;
static int espMode = 0; // 0 = sensor, 1 = vision

// Helper: compute an ImGui display size preserving aspect ratio
static ImVec2 getDisplaySize(const Mat &mat, float maxW, float maxH)
{
    if (mat.empty())
        return ImVec2(0, 0);
    float iw = (float)mat.cols;
    float ih = (float)mat.rows;
    float scale = std::min(maxW / iw, maxH / ih);
    if (scale > 1.0f)
        scale = 1.0f; // don't upscale
    return ImVec2(iw * scale, ih * scale);
}

// Helper: upload/update an OpenGL texture from a cv::Mat (RGB or grayscale)
static GLuint matToTexture(const Mat &mat, GLuint &tex)
{
    if (mat.empty())
        return 0;

    Mat image;
    if (mat.channels() == 1)
        cvtColor(mat, image, COLOR_GRAY2RGB);
    else
        cvtColor(mat, image, COLOR_BGR2RGB);

    if (tex == 0)
        glGenTextures(1, &tex);

    glBindTexture(GL_TEXTURE_2D, tex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.cols, image.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, image.data);
    glBindTexture(GL_TEXTURE_2D, 0);

    return tex;
}

// variabelen voor het berekenen van posities en afstanden
int cxRed = 0;     // X-coördinaat van het middelpunt van de rode contour
int cyRed = 0;     // Y-coördinaat van het middelpunt van de rode contour
int dxRed = 0;     // Verschil in X-coördinaat tussen rode contour en eerste blauwe contour
int dyRed = 0;     // Verschil in Y-coördinaat tussen rode contour en eerste blauwe contour
int LengthRed = 0; // Afstand tussen rode contour en eerste blauwe contour in pixels

int cxBlue1 = 0;    // X-coördinaat van het middelpunt van de eerste blauwe contour
int cyBlue1 = 0;    // Y-coördinaat van het middelpunt van de eerste blauwe contour
int cxBlue2 = 0;    // X-coördinaat van het middelpunt van de tweede blauwe contour
int cyBlue2 = 0;    // Y-coördinaat van het middelpunt van de tweede blauwe contour
int dxBlue = 0;     // Verschil in X-coördinaat tussen eerste en tweede blauwe contour
int dyBlue = 0;     // Verschil in Y-coördinaat tussen eerste en tweede blauwe contour
int LengthBlue = 0; // Afstand tussen eerste en tweede blauwe contour in pixels

// Positie en regelparameters
double ActualPosition = 0.0;   // Actuele positie van de knikker in mm (0-200 mm)
double DesiredPosition = 0.0;  // Gewenste positie van de knikker in mm (0-200 mm)
double proportionalGain = 0.0; // wordt elke iteratie geüpdatet
double DiffValue = 0.6;        // Standaard waarde

// Trackbar instellingen voor de rode mask
int redHighH1 = 10;
int redLowH2 = 170;
int redLowS = 120;
int redHighS = 255;
int redLowV = 50;
int redHighV = 255;

// Trackbar instellingen voor de blauwe border-masker
int blueLowH = 85;
int blueHighH = 99;
int blueHighS = 255;
int blueLowS = 40;
int blueLowV = 200;
int blueHighV = 255;

// Proportionele versterking (hoog-res trackbar: schaal 0.001)
float desiredPositionFloat = 100.0f;  // Gewenste positie als trackbarwaarde in mm
float proportionalGainFloat = 0.035f; // 1.000 -> waarde = proportionalGainFloat / 1000.0
float DiffValueFloat = 0.6f;          // Standaard waarde als trackbarwaarde (x0.01)

void pushValues(int serial_port);

int main()
{

    cout << "OpenCV version: " << CV_VERSION << endl;

    // Open default camera
    VideoCapture cap(0, cv::CAP_V4L2); // Gebruik V4L2 backend (Linux) - pas aan indien nodig voor andere platforms

    // Check of camera kan worden geopend ...
    if (!cap.isOpened())
    {
        cerr << "Can't open camera" << endl;
        return -1;
    }

    // Open de seriële poort
    int serial_port = open("/dev/ttyUSB0", O_RDWR);

    // Check of de seriële poort succesvol is geopend
    if (serial_port < 0)
    {
        perror("Error opening serial port");
        return -1;
    }

    if (!glfwInit())
        return -1;

    GLFWwindow *window =
        glfwCreateWindow(800, 600,
                         "Knikkerbaan GUI",
                         nullptr,
                         nullptr);

    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    // Configureer de seriële poort (baudrate, data bits, stop bits, parity, etc.)
    termios tty;                  // Struct voor seriële poort instellingen
    tcgetattr(serial_port, &tty); // Lees huidige instellingen

    // Stel de baudrate in (115200 baud)
    cfsetispeed(&tty, B115200); // Stel de baudrate in (115200 baud)
    cfsetospeed(&tty, B115200); // Stel de baudrate in (115200 baud)

    // Configureer de seriële poort instellingen (8N1, geen flow control)
    tty.c_cflag |= (CLOCAL | CREAD); // Enable receiver, set local mode
    tty.c_cflag &= ~CSIZE;           // Clear current data bits setting
    tty.c_cflag |= CS8;              // Set 8 data bits
    tty.c_cflag &= ~PARENB;          // No parity
    tty.c_cflag &= ~CSTOPB;          // 1 stop bit
    tty.c_cflag &= ~CRTSCTS;         // No hardware flow control

    // Stel de nieuwe instellingen in
    tcsetattr(serial_port, TCSANOW, &tty);

    // Instructies voor de gebruiker op de console
    cout << "\n=== Vision knikkerbaan ===" << endl;
    cout << getBuildInformation() << endl;
    cout << "Klik op de Push-knop in de GUI om waarden naar de microcontroller te sturen" << endl;
    cout << "ESC: Afsluiten" << endl;

    // Main loop
    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        int display_w, display_h;
        glfwGetFramebufferSize(
            window,
            &display_w,
            &display_h);

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(
            ImVec2((float)display_w,
                   (float)display_h));

        ImGui::Begin(
            "MainWindow",
            nullptr,
            ImGuiWindowFlags_NoTitleBar |
                ImGuiWindowFlags_NoResize |
                ImGuiWindowFlags_NoMove |
                ImGuiWindowFlags_NoCollapse);

        // Homepage
        if (currentScreen == SCREEN_HOME)
        {
            ImGui::SetCursorPosY(80);
            ImGui::Text("Knikkerbaan Regelaar");
            ImGui::Spacing();
            ImGui::Spacing();

            ImGui::Text("Status:");
            ImGui::Text("Desired Position: %.1f mm", DesiredPosition);
            ImGui::Text("Actual Position: %.1f mm", ActualPosition);
            ImGui::Text("Kp: %.3f", proportionalGain);
            ImGui::Text("Td: %.2f s", DiffValue);

            ImGui::Spacing();
            ImGui::Spacing();
            ImGui::Spacing();

            ImGui::SetCursorPosX((display_w - 200) / 2.0f);
            if (ImGui::Button("Thresholds", ImVec2(200, 40)))
            {
                currentScreen = SCREEN_THRESHOLD;
            }

            ImGui::Spacing();

            ImGui::SetCursorPosX((display_w - 200) / 2.0f);
            if (ImGui::Button("Communicatie", ImVec2(200, 40)))
            {
                currentScreen = SCREEN_COMM;
            }
        }
        // Threshold-scherm
        else if (currentScreen == SCREEN_THRESHOLD)
        {
            ImGui::SetCursorPosY(20);
            ImGui::Text("Threshold Instellingen");
            ImGui::Separator();

            // Layout: sliders on the left, big image on the right
            ImGui::Columns(2);
            ImGui::SetColumnWidth(0, 380.0f);

            // Left column: sliders and controls
            ImGui::Text("Rood Masker");
            ImGui::SliderInt("Red H min##red2", &redLowH2, 0, 180);
            ImGui::SliderInt("Red H max##red1", &redHighH1, 0, 180);
            ImGui::SliderInt("Red S min##red", &redLowS, 0, 255);
            ImGui::SliderInt("Red S max##red", &redHighS, 0, 255);
            ImGui::SliderInt("Red V min##red", &redLowV, 0, 255);
            ImGui::SliderInt("Red V max##red", &redHighV, 0, 255);

            ImGui::Text("Blauw Masker");
            ImGui::SliderInt("Blue H min##blue", &blueLowH, 0, 179);
            ImGui::SliderInt("Blue H max##blue", &blueHighH, 0, 179);
            ImGui::SliderInt("Blue S min##blue", &blueLowS, 0, 255);
            ImGui::SliderInt("Blue S max##blue", &blueHighS, 0, 255);
            ImGui::SliderInt("Blue V min##blue", &blueLowV, 0, 255);
            ImGui::SliderInt("Blue V max##blue", &blueHighV, 0, 255);

            ImGui::Spacing();
            ImGui::Checkbox("Autoscale images", &autoscaleImages);
            const char *items = "Rood Masker\0Blauw Masker\0Overlay\0";
            ImGui::Combo("View", &viewMode, items);

            ImGui::Spacing();
            ImGui::SetCursorPosX((ImGui::GetColumnWidth() - 200) / 2.0f);
            if (ImGui::Button("Terug", ImVec2(200, 40)))
            {
                currentScreen = SCREEN_HOME;
            }

            ImGui::NextColumn();

            // Right column: big image view
            Mat viewMat;
            if (viewMode == 0)
            {
                if (!mask.empty())
                {
                    viewMat = mask;
                    matToTexture(viewMat, texMask);
                }
            }
            else if (viewMode == 1)
            {
                if (!mask3.empty())
                {
                    viewMat = mask3;
                    matToTexture(viewMat, texMaskBlue);
                }
            }
            else if (viewMode == 2)
            {
                if (!src.empty() && !mask.empty())
                {
                    Mat colored = Mat::zeros(src.size(), CV_8UC3);
                    colored.setTo(Scalar(0, 0, 255), mask); // red overlay where mask is set
                    Mat overlay;
                    addWeighted(src, 1.0, colored, 0.5, 0.0, overlay);
                    viewMat = overlay;
                    matToTexture(viewMat, texOverlay);
                }
            }

            ImGui::BeginChild("ImageView", ImVec2(0, 0), false);
            if (!viewMat.empty())
            {
                ImGui::Text("View");
                ImVec2 avail = ImGui::GetContentRegionAvail();
                float maxW = avail.x;
                float maxH = (float)display_h - 120.0f; // allow tall view
                ImVec2 sz = autoscaleImages ? getDisplaySize(viewMat, maxW, maxH) : ImVec2(640, 480);
                GLuint whichTex = 0;
                if (viewMode == 0)
                    whichTex = texMask;
                else if (viewMode == 1)
                    whichTex = texMaskBlue;
                else if (viewMode == 2)
                    whichTex = texOverlay;
                ImGui::Image((void *)(intptr_t)whichTex, sz);
            }
            else
            {
                ImGui::Text("View: niet beschikbaar");
            }
            ImGui::EndChild();

            ImGui::Columns(1);
        }
        // Communicatie-scherm
        else if (currentScreen == SCREEN_COMM)
        {
            ImGui::SetCursorPosY(20);
            ImGui::Text("Communicatie Instellingen");
            ImGui::Separator();

            ImGui::Text("Regelparameters");
            ImGui::SliderFloat("Desired Position [mm]",
                               &desiredPositionFloat,
                               0.0f,
                               200.0f);
            ImGui::SliderFloat("Kp",
                               &proportionalGainFloat,
                               0.02f,
                               0.06f,
                               "%.3f");
            ImGui::SliderFloat("Td [s]",
                               &DiffValueFloat,
                               0.3f,
                               0.9f,
                               "%.2f");

            ImGui::Separator();
            const char *espItems = "Sensor\0Vision\0";
            ImGui::Combo("ESP32 Mode", &espMode, espItems);
            ImGui::Spacing();

            ImGui::SetCursorPosX((display_w - 200) / 2.0f);
            if (ImGui::Button("Push", ImVec2(200, 40)))
            {
                pushValues(serial_port);
            }

            ImGui::SetCursorPosX((display_w - 200) / 2.0f);
            if (ImGui::Button("Terug", ImVec2(200, 40)))
            {
                currentScreen = SCREEN_HOME;
            }
        }

        ImGui::End();

        ImGui::Render();

        glViewport(
            0,
            0,
            display_w,
            display_h);

        glClearColor(
            0.1f,
            0.1f,
            0.1f,
            1.0f);

        glClear(GL_COLOR_BUFFER_BIT);

        ImGui_ImplOpenGL3_RenderDrawData(
            ImGui::GetDrawData());

        glfwSwapBuffers(window);

        // 1. Frame van camera lezen
        cap.read(src);

        // 1. BGR → HSV
        cvtColor(src, hsv, COLOR_BGR2HSV);

        // 2. Mask voor rood (2 ranges!)
        inRange(hsv, Scalar(0, redLowS, redLowV), Scalar(redHighH1, redHighS, redHighV), mask1);  // Lage rood range
        inRange(hsv, Scalar(redLowH2, redLowS, redLowV), Scalar(180, redHighS, redHighV), mask2); // Hoge rood range

        // inRange(hsv, Scalar(0, 0, 215), Scalar(180, 80, 255), mask3);   // Geel range
        if (blueLowH > blueHighH)
            std::swap(blueLowH, blueHighH);
        if (blueLowS > blueHighS)
            std::swap(blueLowS, blueHighS);
        if (blueLowV > blueHighV)
            std::swap(blueLowV, blueHighV);
        inRange(hsv, Scalar(blueLowH, blueLowS, blueLowV), Scalar(blueHighH, blueHighS, blueHighV), mask3); // Blauw range

        // Combineer de twee rood ranges in één mask
        mask = mask1 | mask2;

        // Ruis verwijderen (rode mask)
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
        morphologyEx(mask, mask, MORPH_OPEN, kernel);
        morphologyEx(mask, mask, MORPH_CLOSE, kernel);

        // Ruis verwijderen (blauwe mask)
        morphologyEx(mask3, mask3, MORPH_OPEN, kernel);  // Openen: eerst erode, dan dilate (verwijdert kleine witte ruis)
        morphologyEx(mask3, mask3, MORPH_CLOSE, kernel); // Sluiten: eerst dilate, dan erode (vult kleine zwarte gaten in de contouren)

        // Contours van rood vinden
        vector<vector<Point>> Ballcontours;                                   // Vector voor rode contours
        findContours(mask, Ballcontours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE); // Alleen externe contours, geen hiërarchie nodig, eenvoudige benadering

        // Contours / connected components van blauw vinden
        Mat labels, stats, centroids;
        int nLabels = connectedComponentsWithStats(mask3, labels, stats, centroids);

        // Verzamel componenten (area, centroid)
        vector<pair<int, Point>> comps; // pair<area, center>
        for (int lbl = 1; lbl < nLabels; ++lbl)
        {
            int area = stats.at<int>(lbl, CC_STAT_AREA);
            if (area > 50)
            { // filter kleine ruis
                int cx = static_cast<int>(centroids.at<double>(lbl, 0));
                int cy = static_cast<int>(centroids.at<double>(lbl, 1));
                comps.emplace_back(area, Point(cx, cy));
            }
        }

        // Sorteer aflopend op oppervlakte
        sort(comps.begin(), comps.end(), [](const pair<int, Point> &a, const pair<int, Point> &b)
             { return a.first > b.first; });

        // Maak een eenvoudige lijst met border-centroids (vul met 0 als niet gevonden)
        vector<Point> borderCenters(2, Point(0, 0));
        for (size_t i = 0; i < comps.size() && i < 2; ++i)
        {
            borderCenters[i] = comps[i].second;
        }

        bool hasTwoBlueBorders = (comps.size() >= 2);
        if (hasTwoBlueBorders)
        {
            line(src, borderCenters[0], borderCenters[1], Scalar(0, 255, 255), 2);
            circle(src, borderCenters[0], 10, Scalar(0, 255, 255), 2);
            circle(src, borderCenters[1], 10, Scalar(0, 255, 255), 2);
        }

        // Check of er überhaupt rode contours zijn gevonden
        if (Ballcontours.empty())
        {
            // continue; // Ga terug naar het begin van de loop als er geen rode contours zijn gevonden (om fouten te voorkomen bij het berekenen van het middelpunt)
        }

        // Check of er überhaupt genoeg blauwe componenten zijn gevonden (minimaal 2 borders nodig)
        if (!hasTwoBlueBorders)
        {
            // continue; // Ga terug naar het begin van de loop als er niet genoeg borders zijn gevonden
        }

        // 5. Grootste contour pakken (aannemen = bal)
        int largestIndexRed = 0;
        double maxAreaRed = 0;

        // Loop door alle contours en vind de grootste
        for (int i = 0; i < Ballcontours.size(); i++)
        {

            double areaRed = contourArea(Ballcontours[i]);

            // Alleen overschrijven als de contour groot genoeg is (om ruis te vermijden)
            if (areaRed > maxAreaRed && areaRed > 500)
            {
                maxAreaRed = areaRed;
                largestIndexRed = i;
            }
        }

        // 6. Middelpunt van de bal berekenen
        Moments mRed = moments(Ballcontours[largestIndexRed]); // Bereken het middelpunt van de grootste rode contour (de bal)

        // Bereken de coördinaten van het middelpunt van de rode contour (de bal)
        cxRed = int(mRed.m10 / mRed.m00);
        cyRed = int(mRed.m01 / mRed.m00);

        // Bereken de coördinaten van het middelpunt van de blauwe contouren (de borders)
        if (borderCenters[0].x < borderCenters[1].x)
        {
            cxBlue1 = borderCenters[0].x;
            cyBlue1 = borderCenters[0].y;
            cxBlue2 = borderCenters[1].x;
            cyBlue2 = borderCenters[1].y;
        }
        else
        {
            cxBlue1 = borderCenters[1].x;
            cyBlue1 = borderCenters[1].y;
            cxBlue2 = borderCenters[0].x;
            cyBlue2 = borderCenters[0].y;
        };

        dyRed = cyRed - cyBlue1;
        dxRed = cxRed - cxBlue1;

        dyBlue = cyBlue1 - cyBlue2;
        dxBlue = cxBlue1 - cxBlue2;

        LengthBlue = sqrt(dxBlue * dxBlue + dyBlue * dyBlue);

        LengthRed = sqrt(dxRed * dxRed + dyRed * dyRed);

        if (LengthBlue < 1)
            LengthBlue = 1;
        if (LengthRed < 1)
            LengthRed = 1;

        // Bereken de actuele positie van de bal in mm (0-200 mm)
        ActualPosition = (static_cast<double>(LengthRed) / static_cast<double>(LengthBlue)) * 200.0; // Actuele positie van de bal in mm

        // Beperk de waarde van ActualPosition tot het bereik 0-200 mm
        if (ActualPosition < 0.0)
            ActualPosition = 0.0;
        if (ActualPosition > 200.0)
            ActualPosition = 200.0;

        // Update desired position vanuit de slider en converteer Kp integer (x0.001) naar double
        DesiredPosition = static_cast<double>(desiredPositionFloat); // Gewenste positie in mm
        proportionalGain = static_cast<double>(proportionalGainFloat);
        DiffValue = static_cast<double>(DiffValueFloat);

        // Stuur de actuele positie van de bal naar de microcontroller via de seriële poort
        string Posmsg = "$Actpos," + to_string(ActualPosition) + "*\n";
        write(serial_port, Posmsg.c_str(), Posmsg.length());

        //========================== Visualisatie ==========================

        // 7. Teken het middelpunt van de bal en de uiteinden van de knikkerbaan
        circle(src, Point(cxRed, cyRed), 5, Scalar(0, 255, 0), -1);   // Groen cirkeltje op het middelpunt
        circle(src, Point(cxBlue1, cyBlue1), 5, Scalar(0, 0, 0), -1); // Zwart cirkeltje op de border
        circle(src, Point(cxBlue2, cyBlue2), 5, Scalar(0, 0, 0), -1); // Zwart cirkeltje op de border

        // Teken de contouren van de bal en de uiteinden van de knikkerbaan
        drawContours(src, Ballcontours, largestIndexRed, Scalar(255, 0, 0), 2); // Blauwe contour van de bal
        // Visualiseer de twee border-centroids en verbind ze
        line(src, borderCenters[0], borderCenters[1], Scalar(0, 255, 255), 2);
        circle(src, borderCenters[0], 10, Scalar(0, 255, 255), 2);
        circle(src, borderCenters[1], 10, Scalar(0, 255, 255), 2);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();

    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    // Camera vrijgeven
    cap.release();

    // Alle vensters sluiten
    destroyAllWindows();

    return 0;
}

// Functie om waarden naar de microcontroller te sturen via de seriële poort
void pushValues(int serial_port)
{
    // Stuur de gewenste positie naar de microcontroller via de seriële poort
    ostringstream ss;                                        // String stream voor het formatteren van de gewenste positie
    ss << fixed << setprecision(1) << DesiredPosition;       // Formatteren naar 1 decimaal
    string Desirmsg = "$Desirpos," + ss.str() + "*\n";       // Maak het bericht voor de gewenste positie
    write(serial_port, Desirmsg.c_str(), Desirmsg.length()); // Stuur het bericht via de seriële poort

    ostringstream ss2;
    ss2 << fixed << setprecision(3) << proportionalGain;
    string Kpmsg = "$Kp," + ss2.str() + "*\n";
    write(serial_port, Kpmsg.c_str(), Kpmsg.length());

    ostringstream ss3;
    ss3 << fixed << setprecision(3) << DiffValue;
    string Diffmsg = "$Td," + ss3.str() + "*\n";
    write(serial_port, Diffmsg.c_str(), Diffmsg.length());

    string modeStr = (espMode == 1) ? "vision" : "sensor";
    string Modemsg = "$Mode," + modeStr + "*\n";
    write(serial_port, Modemsg.c_str(), Modemsg.length());

    cout << "Pushed values to microcontroller:" << endl;
    cout << Kpmsg << endl;
    cout << Diffmsg << endl;
    cout << Modemsg << endl;
}
