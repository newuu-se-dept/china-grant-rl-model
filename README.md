# china-grant-rl-model

## Usage NeTrainSim on Mac
Install `qt` and `cmake`:
```
brew install qt
brew install cmake
```
Build Mac executable
```
cd NeTrainSim-adjusted
./build-mac.sh
```
Run the sample project:
```
mkdir res
./build-mac/src/NeTrainSimConsole/NeTrainSim -n src/data/sampleProject/nodesFile.dat -l src/data/sampleProject/linksFile.dat -t src/data/sampleProject/dieselTrain.dat -o res
```
