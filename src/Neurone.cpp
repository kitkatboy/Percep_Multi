#include "Neurone.h"


Neurone::Neurone() {
    entree = 0.0;
    sortie = 0.0;
    gradient = 0.0;
}


Neurone::~Neurone() {
}


void Neurone::set_entree(double val) {
    entree = val;
}


double Neurone::get_entree() {
    return entree;
}


double Neurone::get_sortie() {
    return sortie;
}


void Neurone::set_sortie(double val) {
    sortie = val;
}


void Neurone::add_poids(double d){
    liste_poids.push_back(d);
}


double Neurone::get_poids(unsigned int i) {
    return liste_poids[i];
}


void Neurone::set_poids(unsigned int i, double p) {
    liste_poids[i] = p;
}


double Neurone::get_gradient() {
    return gradient;
}


void Neurone::set_gradient(double val) {
    gradient = val;
}


void Neurone::calcul_gradient() {
    gradient =  sortie * (1 - sortie) * (entree - sortie);
}