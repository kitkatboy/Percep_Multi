#include <cv.h>
#include "Extractor.h"
#include "Mlp.h"


int main(int argc, char *argv[])
{
    Mat* app = new Mat(imread("data/app.tif", 0));
    Mat* test = new Mat(imread("data/test.tif", 0));
    std::vector< std::vector<double> > *examples_features, *tests_features;


    // Display source
//    namedWindow("Source", WINDOW_NORMAL);
//    imshow("Source",*app);
//    waitKey(0);


    // Extraction densites zoning
    Extractor * extract = new Extractor();

    extract->set_data(app);
    extract->extraction(std::pair<int,int>(0,0), std::pair<int,int>(0,0), 0, new Mat());
    examples_features = extract->get_densites();

    extract->set_data(test);
    extract->extraction(std::pair<int,int>(0,0), std::pair<int,int>(0,0), 0, new Mat());
    tests_features = extract->get_densites();

    delete extract;


    // Benchmarks bash arguments
//    std::string tmp = argv[1];
//    int neurones = atoi(tmp.c_str());
//    std::string tmp2 = argv[2];
//    double error = atof(tmp2.c_str());


    // Perceptron multicouches
    Mlp *reseau = new Mlp();
    reseau->run(examples_features, tests_features, 28, 1.2);            // Best results
//    reseau->run(examples_features, tests_features, neurones, error);    // for Benchmarks

    delete reseau;


    delete examples_features;
    delete tests_features;
    delete app;
    delete test;

    return 0;
}