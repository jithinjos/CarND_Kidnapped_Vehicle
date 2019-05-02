/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::normal_distribution;
using std::numeric_limits;
using std::string;
using std::uniform_int_distribution;
using std::uniform_real_distribution;
using std::vector;

void ParticleFilter::init(double x,
                          double y,
                          double theta,
                          double std[])
{
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */

  if (is_initialized)
  {
    return;
  }

  num_particles = 100; // TODO: Set the number of particles

  std::default_random_engine gen;

  // Get standard deviations
  double std_x = std[0];
  double std_y = std[1];
  double std_theta = std[2];

  // Create normal distributions for x, y and theta
  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);

  // Generate particles with normal distribution with mean as GPS values.
  for (int i = 0; i < num_particles; i++)
  {

    Particle particle;
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1.0;

    particles.push_back(particle);
  }

  //  Set "initialized" status.
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t,
                                double std_pos[],
                                double velocity,
                                double yaw_rate)
{
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine gen;

  // Get standard deviations
  double std_x = std_pos[0];
  double std_y = std_pos[1];
  double std_theta = std_pos[2];

  // Create normal distributions for x, y and theta around 0
  normal_distribution<double> dist_x(0, std_x);
  normal_distribution<double> dist_y(0, std_y);
  normal_distribution<double> dist_theta(0, std_theta);

  for (int i = 0; i < num_particles; i++)
  {
    double min_yaw_rate = 0.00001;

    double theta = particles[i].theta;
    if (fabs(yaw_rate) >= min_yaw_rate)
    {
      particles[i].x += velocity / yaw_rate * (sin(theta + yaw_rate * delta_t) - sin(theta));
      particles[i].y += velocity / yaw_rate * (cos(theta) - cos(theta + yaw_rate * delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }
    else
    {
      particles[i].x += velocity * cos(theta) * delta_t;
      particles[i].y += velocity * sin(theta) * delta_t;
    }

    // Add gaussian noise.
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs> &observations)
{
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */

  for (unsigned int i = 0; i < observations.size(); i++) // For each observation
  {
    // Initialize min distance as a really big number.
    double min_distance = numeric_limits<double>::max();

    // Initialize the found map to something not possible.
    int mapId = -1;

    for (unsigned j = 0; j < predicted.size(); j++) // For each predition.
    {
      double x_distance = observations[i].x - predicted[j].x;
      double y_distance = observations[i].y - predicted[j].y;
      double distance = (x_distance * x_distance) + (y_distance * y_distance);

      // If the "distance" is less than minimum, stored the id and update min.
      if (distance < min_distance)
      {
        min_distance = distance;
        mapId = predicted[j].id;
      }
    }
    // Update the observation identifier with the most closely associated prediction.
    observations[i].id = mapId;
  }
}

void ParticleFilter::updateWeights(double sensor_range,
                                   double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks)
{
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  // for each particle...
  for (int i = 0; i < num_particles; i++)
  {

    // particle x, y coordinates
    double p_x = particles[i].x;
    double p_y = particles[i].y;
    double p_theta = particles[i].theta;

    // create a vector to hold the map landmark locations predicted to be within sensor range of the particle
    vector<LandmarkObs> predictions;

    double sensor_range_2 = sensor_range * sensor_range;

    // for each map landmark...
    for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++)
    {

      // get id and x,y coordinates
      float lm_x = map_landmarks.landmark_list[j].x_f;
      float lm_y = map_landmarks.landmark_list[j].y_f;
      int lm_id = map_landmarks.landmark_list[j].id_i;

      double dX = p_x - lm_x;
      double dY = p_y - lm_y;

      // filter out valid predictions
      if (dX * dX + dY * dY <= sensor_range_2)
      {
        predictions.push_back(LandmarkObs{lm_id, lm_x, lm_y});
      }
    }

    // create and populate a copy of the list of observations transformed from vehicle coordinates to map coordinates
    vector<LandmarkObs> tr_observations;
    for (unsigned int j = 0; j < observations.size(); j++)
    {
      double t_x = cos(p_theta) * observations[j].x - sin(p_theta) * observations[j].y + p_x;
      double t_y = sin(p_theta) * observations[j].x + cos(p_theta) * observations[j].y + p_y;
      tr_observations.push_back(LandmarkObs{observations[j].id, t_x, t_y});
    }

    // perform dataAssociation for the predictions and transformed observations on current particle
    // find closest match
    dataAssociation(predictions, tr_observations);

    // reset weight
    particles[i].weight = 1.0;

    for (unsigned int j = 0; j < tr_observations.size(); j++)
    {

      // placeholders for observation and associated prediction coordinates
      double ob_x, ob_y, pr_x, pr_y;
      ob_x = tr_observations[j].x;
      ob_y = tr_observations[j].y;

      int associated_prediction = tr_observations[j].id;

      // get the x,y coordinates of the prediction associated with the current observation
      for (unsigned int k = 0; k < predictions.size(); k++)
      {
        if (predictions[k].id == associated_prediction)
        {
          pr_x = predictions[k].x;
          pr_y = predictions[k].y;
        }
      }

      // calculate weight for this observation with multivariate Gaussian
      double s_x = std_landmark[0];
      double s_y = std_landmark[1];
      double obs_w = (1 / (2 * M_PI * s_x * s_y)) *
                     exp(-(pow(pr_x - ob_x, 2) / (2 * pow(s_x, 2)) + (pow(pr_y - ob_y, 2) / (2 * pow(s_y, 2)))));

      // product of this obersvation weight with total observations weight
      particles[i].weight *= obs_w;
    }
  }
}

void ParticleFilter::resample()
{
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  // Get weights and max weight.
  vector<double> weights;
  double maxWeight = numeric_limits<double>::min();
  for (int i = 0; i < num_particles; i++)
  {
    weights.push_back(particles[i].weight);
    if (particles[i].weight > maxWeight)
    {
      maxWeight = particles[i].weight;
    }
  }

  // Creating distributions.
  uniform_real_distribution<double> distDouble(0.0, maxWeight);
  uniform_int_distribution<int> distInt(0, num_particles - 1);

  // Generating index.
  std::default_random_engine gen;
  int index = distInt(gen);

  double beta = 0.0;

  // resmapling using the wheel
  vector<Particle> reParticles;
  for (int i = 0; i < num_particles; i++)
  {
    beta += distDouble(gen) * 2.0;
    while (beta > weights[index])
    {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    reParticles.push_back(particles[index]);
  }

  particles = reParticles;
}

void ParticleFilter::SetAssociations(Particle &particle,
                                     const vector<int> &associations,
                                     const vector<double> &sense_x,
                                     const vector<double> &sense_y)
{
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord)
{
  vector<double> v;

  if (coord == "X")
  {
    v = best.sense_x;
  }
  else
  {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1); // get rid of the trailing space
  return s;
}
