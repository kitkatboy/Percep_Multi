#ifndef MLP_H
#define MLP_H


#include <iostream>
#include <vector>
#include <algorithm>
#include <time.h>
#include <fstream>
#include "Neurone.h"


class Mlp
{
protected:
    std::vector<Neurone> couche_entree;
    std::vector<Neurone> couche_cachee;
    std::vector<Neurone> couche_sortie;

public:
    Mlp();
    ~Mlp();
    void run(std::vector< std::vector<double> >* examples_features, std::vector< std::vector<double> >* tests_features, int neurones_caches, double learning_err);
    void create(int neurones_caches, unsigned long data_size);
    void learning(std::vector< std::vector<double> >* examples_features, double learning_err);
    void propagation(int choice = 0);
    void retro_propagation();
    void delta_rule(int choice = 0);
    void write_results(int neurones_caches, double learning_err, double time, int errors);
    int test(std::vector< std::vector<double> >* tests_features);
};


#endif