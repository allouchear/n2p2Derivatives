// n2p2 - A neural network potential package
// Copyright (C) 2018 Andreas Singraber (University of Vienna)
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#ifndef TRAINING_H
#define TRAINING_H

#include "DatasetES.h"
#include "Updater.h"
#include <cstddef> // std::size_t
#include <fstream> // std::ofstream
#include <map>     // std::map
#include <random>  // std::mt19937_64
#include <string>  // std::string
#include <vector>  // std::vector

namespace nnp
{

/// Training methods.
class Training : public Dataset
{
public:
    /// Type of update routine.
    enum UpdaterType
    {
        /// Simple gradient descent methods.
        UT_GD,
        /// Kalman filter-based methods.
        UT_KF,
        /// Levenberg-Marquardt algorithm.
        UT_LM
    };

    /** Training parallelization mode.
     *
     * This mode determines if and how individual MPI tasks contribute to
     * parallel training. Note that in all cases the data set gets distributed
     * among the MPI processes and RMSE computation is always parallelized.
     */
    enum ParallelMode
    {
        /** No training parallelization, only data set distribution.
         *
         * Data set is distributed via MPI, but for each weight update only
         * a single task is active, selects energy update candidates,
         * computes errors and gradients, and updates weights.
         */
        //PM_DATASET,
        /** Parallel gradient computation, update on rank 0.
         *
         * Data set is distributed via MPI, each tasks selects energy
         * update candidates, and computes errors and gradients, which are
         * collected on rank 0. Weight update is carried out on rank 0 and new
         * weights are redistributed to all tasks.
         */
        PM_TRAIN_RK0,
        /** Parallel gradient computation, update on each task.
         *
         * Data set is distributed via MPI, each tasks selects energy
         * update candidates, and computes errors and gradients, which are
         * collected on all MPI tasks. Identical weight updates are carried out
         * on each task. This mode is ideal if the update routine itself is
         * parallelized.
         */
        PM_TRAIN_ALL
    };

    /// Jacobian matrix preparation mode.
    enum JacobianMode
    {
        /// No Jacobian, sum up contributions from update candidates.
        JM_SUM,
        /// Prepare one Jacobian entry for each task, sum up within tasks.
        JM_TASK,
        /// Prepare full Jacobian matrix.
        JM_FULL
    };

    /// Update strategies available for Training.
    enum UpdateStrategy
    {
        /// One combined updater for all elements.
        US_COMBINED,
        /// Separate updaters for individual elements.
        US_ELEMENT
    };

    /// How update candidates are selected during Training.
    enum SelectionMode
    {
        /// Select candidates randomly.
        SM_RANDOM,
        /// Sort candidates according to their RMSE and pick worst first.
        SM_SORT,
        /// Select candidates randomly with RMSE above threshold.
        SM_THRESHOLD
    };

    /// Constructor.
    Training();
    /// Destructor, updater vector needs to be cleaned.
    ~Training();
    /** Randomly select training and test set structures.
     *
     * Also fills training candidates lists.
     */
    void                  selectSets();
    /** Write training and test set to separate files (train.data and
     * test.data, same format as input.data).
     */
    void                  writeSetsToFiles();
    /** Initialize weights for all elements.
     */
    void                  initializeWeights();
    /** Initialize weights vector according to update strategy.
     *
     * @param[in] updateStrategy Determines the shape of the weights array.
     */
    void                  initializeWeightsMemory(UpdateStrategy updateStrategy
                                                      = US_COMBINED);
    /** General training settings and setup of weight update routine.
     */
    void                  setupTraining();
    /** Calculate neighbor lists for all structures.
     */
    void                  calculateNeighborLists();
    /** Calculate RMSE for all structures.
     *
     * @param[in] writeCompFiles Write NN and reference charges  to
     *                           comparison files.
     * @param[in] identifier String added to "CHARGE" log line.
     * @param[in] fileNameESTrain File name for training charges
     *                                  comparison file.
     * @param[in] fileNameESTest File name for test charges comparison
     *                                 file.
     */
    void                  calculateRmseES(bool const        writeCompFiles,
                                        std::string const identifier           
                                            = "",
                                        std::string const fileNameESTrain
                                            = "charges-train.comp",
                                        std::string const fileNameESTest
                                            = "charges-test.comp"
                                            );
    /** Calculate RMSE per epoch for all structures with file names used in
     * training loop.
     *
     * Also write training curve to file.
     */
    void                  calculateRmseEpoch();
    /** Write weights to files (one file for each element).
     *
     * @param[in] fileNameFormat String with file name format.
     */
    void                  writeWeights(std::string const fileNameFormat
                                          = "weights.%03zu.data") const;
    /** Write weights to files during training loop.
     */
    void                  writeWeightsEpoch() const;
    /** Write current RMSEs and epoch information to file.
     *
     * @param[in] append If true, append to file, otherwise create new file.
     * @param[in] fileName File name for learning curve file.
     */
    void                  writeLearningCurve(bool              append,
                                             std::string const fileName
                                                 = "learning-curve.out") const;
    /** Write neuron statistics collected since last invocation.
     *
     * @param[in] fileName File name for statistics file.
     */
    void                  writeNeuronStatistics(std::string const fileName
                                                   = "neuron-stats.out") const;
    /** Write neuron statistics during training loop.
     */
    void                  writeNeuronStatisticsEpoch() const;
    /** Reset neuron statistics for all elements.
     */
    void                  resetNeuronStatistics() const;
    /** Write updater information to file.
     *
     * @param[in] append If true, append to file, otherwise create new file.
     * @param[in] fileNameFormat String with file name format.
     */
    void                  writeUpdaterStatus(bool              append,
                                             std::string const fileNameFormat
                                                 = "updater.%03zu.out") const;
    /** Sort update candidates with descending RMSE.
     */
    void                  sortUpdateCandidates();
    /** Shuffle update candidates.
     */
    void                  shuffleUpdateCandidates();
    /** Check if selection mode should be changed.
     */
    void                  checkSelectionMode();
    /** Execute main training loop.
     */
    void                  loop();
    /** Perform one update.
     *
     * @param[in] otherwise energy update.
     */
    void                  update();
    /** Get a single weight value.
     *
     * @param[in] element Element index of weight.
     * @param[in] index Weight index.
     *
     * @return Weight value.
     *
     * Note: This function is implemented for testing purposes and works
     * correctly only with update strategy #US_ELEMENT.
     */
    double                getSingleWeight(std::size_t element,
                                          std::size_t index);
    /** Set a single weight value.
     *
     * @param[in] element Element index of weight.
     * @param[in] index Weight index.
     * @param[in] value Weight value.
     *
     * Note: This function is implemented for testing purposes and works
     * correctly only with update strategy #US_ELEMENT.
     */
    void                  setSingleWeight(std::size_t element,
                                          std::size_t index,
                                          double      value);
    /** Calculate derivatives of energy with respect to weights.
     *
     * @param[in,out] structure Structure to process.
     *
     * @return Vector with derivatives of energy with respect to weights (per
     *         element).
     *
     * @note This function is implemented for testing purposes.
     */
    std::vector<
    std::vector<double> > calculateWeightDerivatives(Structure* structure);
    /** Set training log file name
     *
     * @param[in] fileName File name for training log.
     */
    void                  setTrainingLogFileName(std::string fileName);

private:
    /// Contains location of one update candidate (energy ).
    struct UpdateCandidate
    {
        /// Structure index.
        std::size_t s;
        /// Absolute value of error with respect to reference value.
        double      error;

        /// Overload < operator to sort in \em descending order.
        bool operator<(UpdateCandidate const& rhs) const;
    };

    /// Charge factor.
    double                   Qfactor;
    /// Dipole factor.
    double                   Dipolefactor;
    /// Updater type used.
    UpdaterType                   updaterType;
    /// Parallelization mode used.
    ParallelMode                  parallelMode;
    /// Jacobian mode used.
    JacobianMode                  jacobianMode;
    /// Update strategy used.
    UpdateStrategy                updateStrategy;
    /// Selection mode for update candidates.
    SelectionMode                 selectionMode;
    /// If this rank performs weight updates.
    bool                          hasUpdaters;
    /// If this rank holds structure information.
    bool                          hasStructures;
    /// Free symmetry function memory after calculation.
    bool                          freeMemory;
    /// Whether training log file is written.
    bool                          writeTrainingLog;
    /// Number of updaters (depends on update strategy).
    std::size_t                   numUpdaters;
    /// Number of energies in training set.
    std::size_t                   numESTrain;
    /// Number of epochs requested.
    std::size_t                   numEpochs;
    /// Batch size for each MPI task (energies).
    std::size_t                   taskBatchSizeES;
    /// Current epoch.
    std::size_t                   epoch;
    /// Write energy comparison every this many epochs.
    std::size_t                   writeESEvery;
    /// Write weights every this many epochs.
    std::size_t                   writeWeightsEvery;
    /// Write neuron statistics every this many epochs.
    std::size_t                   writeNeuronStatisticsEvery;
    /// Up to this epoch energy comparison is written every epoch.
    std::size_t                   writeESAlways;
    /// Up to this epoch weights are written every epoch.
    std::size_t                   writeWeightsAlways;
    /// Up to this epoch neuron statistics are written every epoch.
    std::size_t                   writeNeuronStatisticsAlways;
    /// Current position in energy update candidate list (SM_SORT).
    std::size_t                   posUpdateCandidatesES;
    /// Maximum trials for SM_THRESHOLD selection mode.
    std::size_t                   rmseThresholdTrials;
    /// Update counter.
    std::size_t                   countUpdates;
    /// Number of energy updates per epoch.
    std::size_t                   esUpdates;
    /// ES used per update.
    std::size_t                   energiesPerUpdate;
    /// ES used per update (summed over all MPI tasks).
    std::size_t                   energiesPerUpdateGlobal;
    /// Global number of energy errors per update.
    std::size_t                   errorsGlobalES;
    /// Total number of weights.
    std::size_t                   numWeights;
    /// Desired energy update fraction per epoch.
    double                        epochFractionES;
    /// Current RMSE of training charges.
    double                        rmseChargesTrain;
    /// Current RMSE of test charges.
    double                        rmseChargesTest;
    /// Current RMSE of training dipoles.
    double                        rmseDipolesTrain[3];
    /// Current RMSE of test dipoles.
    double                        rmseDipolesTest[3];
    /// RMSE threshold for energy update candidates.
    double                        rmseThresholdES;
    /// Force update weight.
    std::string                   trainingLogFileName;
    /// Training log file.
    std::ofstream                 trainingLog;
    /// Errors per task for each energy update.
    std::vector<int>              errorsPerTaskES;
    /// Offset for combined energy error per task.
    std::vector<int>              offsetPerTaskES;
    /// Number of weights per updater.
    std::vector<std::size_t>      numWeightsPerUpdater;
    /// Offset of each element's weights in combined array.
    std::vector<std::size_t>      weightsOffset;
    /// Vector with indices of training structures.
    std::vector<UpdateCandidate>  updateCandidatesES;
    /// Weights per task per updater for energy updates.
    std::vector<
    std::vector<int> >            weightsPerTaskES;
    /// Stride for Jacobians per task per updater for energy updates.
    std::vector<
    std::vector<int> >            offsetJacobianES;
    /// Neural network weights and biases for each element.
    std::vector<
    std::vector<double> >         weights;
    /// Global error vector for energies (per updater).
    std::vector<
    std::vector<double> >         errorE;
    /// Global Jacobian for energies (per updater).
    std::vector<
    std::vector<double> >         jacobianE;
    /// Weight updater (combined or for each element).
    std::vector<Updater*>         updaters;
    /// Schedule for varying selection mode.
    std::map<std::size_t,
             SelectionMode>       selectionModeSchedule;
    /// Per-task random number generator.
    std::mt19937_64               rngNew;
    /// Global random number generator.
    std::mt19937_64               rngGlobalNew;

    /** Check if training loop should be continued.
     *
     * @return True if further training should be performed, false otherwise.
     */
    bool advance() const;
    /** Get weights from neural network class.
     */
    void getWeights();
    /** Set weights in neural network class.
     */
    void setWeights();
    /** Write energy update data to training log file.
     *
     * @param[in] proc Processor which provided update candidate.
     * @param[in] il Loop index of threshold loop.
     * @param[in] f RMSE fraction of update candidate.
     * @param[in] is Local structure index.
     * @param[in] isg Global structure index.
     */
    void addTrainingLogEntry(int                 proc,
                             std::size_t         il,
                             double              f,
                             std::size_t         isg,
                             std::size_t         is);
};

//////////////////////////////////
// Inlined function definitions //
//////////////////////////////////

inline bool Training::UpdateCandidate::operator<(
                                    Training::UpdateCandidate const& rhs) const
{
    return this->error > rhs.error;
}

}

#endif
