#ifndef EXTRACTOR_H
#define EXTRACTOR_H


#include <cv.h>
#include <highgui.h>


using namespace cv;

class Extractor
{
protected:
    Mat *img;
    std::vector< std::pair<int,int> > to_write;
    std::vector< std::vector<double> > *features = new std::vector< std::vector<double> >;

public:
    Extractor();
    ~Extractor();
    void set_data(Mat* data);
    void extraction(std::pair<int,int> haut_g, std::pair<int,int> bas_d, int choice, Mat* tmp);
    void zoning(std::pair<int,int> haut_g, std::pair<int,int> bas_d);
    std::vector< std::vector<double> >* get_densites();
    void show_element(int line, int column);
    void show_histo(int choice);
};


#endif