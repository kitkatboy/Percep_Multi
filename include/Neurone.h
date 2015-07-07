#ifndef NEURONE_H
#define NEURONE_H


#include <vector>


class Neurone
{
protected:
    double entree;                      // Si le neurone appartient à la couche de sortie, ce parametre correspond a la sortie attendue.
    double sortie;
    double gradient;                    // gradient d'erreur
    std::vector<double> liste_poids;    // Liste des poids des arcs vers les neurones de la couche suivante

public:
    Neurone();
    ~Neurone();
    void set_entree(double val);
    double get_entree();
    double get_sortie();
    void set_sortie(double val);
    void add_poids(double);
    double get_poids(unsigned int);
    void set_poids(unsigned int, double);
    double get_gradient();
    void set_gradient(double val);
    void calcul_gradient();
};


#endif // NEURONE_H