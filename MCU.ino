#include <EloquentTinyML.h> /*tensor model converter*/

#include "arduinoFFT.h" /*fft library*/
#include "sound_model.h" /*custom model*/

/*about model features*/
#define N_INPUTS 128
#define N_OUTPUTS 10
#define TENSOR_ARENA_SIZE 16*1024
#define Mic A0

Eloquent::TinyML::TfLite<N_INPUTS, N_OUTPUTS, TENSOR_ARENA_SIZE> nn; /*create model object*/

/*about sampling*/
const float sampleWindow = 3850;
int input;
unsigned int sample;
unsigned long starts;

/*about fft*/
const uint16_t samples = 256;
const double signalFrequency = 256;
const double samplingFrequency = 256;
double temp[512];
double data[256];
float fft[128];
String sound[10] = {"air conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
                    "engine_idling", "gun_shot", "jackhammer", "siren", "street_music"};

void setup() {
  Serial.begin(2000000);
  nn.begin(sound_model);
  Serial.println("1. Get data for 1 second");
  Serial.println("2. Check data for 1 second");
  Serial.println("3. FFT the Data");
  Serial.println("4. Predict through  the Data");
}

void loop() {
  if(Serial.available()>0) {
    input = Serial.read();
      if(input == '1'){
      starts = micros();
      for(int i = 0; i<512; i++){
        unsigned long t0 = micros();
        unsigned int peakToPeak = 0;
        unsigned int signalMax = 0;
        unsigned int signalMin = 1024;
        while (micros() - t0 < sampleWindow){
            sample = analogRead(Mic);
            if (sample < 1024){
              if (sample > signalMax)
             {
                signalMax = sample;  // save just the max levels
             }
              else if (sample < signalMin)
             {
                signalMin = sample;  // save just the min levels
             }
            }
          }
          peakToPeak = signalMax - signalMin;  // max - min = peak-peak amplitude
          double convert = peakToPeak*(1.0)  / 1024;  // convert like training data
          temp[i] = convert;
          Serial.println(convert);
      }
      Serial.println("Time spend : " + String(micros()-starts));

      for(int i =0; i<256; i++){
        data[i] = temp[i+128];   
      }
    }
    else if(input=='2'){
      for(int i = 0; i <256; i++){
        Serial.println(data[i]);
      }
    }
    else if(input=='3'){
      arduinoFFT FFT = arduinoFFT(); /* Create FFT object */
      double vImag[256] = {0,};
      
      Serial.print("weigh data ...");
      FFT.Windowing(data, samples, FFT_WIN_TYP_HAMMING, FFT_FORWARD); /* Weigh data */
      Serial.println("completed");
      Serial.print("Compute FFT of data ...");
      FFT.Compute(data, vImag, samples, FFT_FORWARD); /* Compute FFT */
      Serial.println("completed");
      Serial.print("Compute magnitudes of FFT ...");
      FFT.ComplexToMagnitude(data, vImag, samples); /* Compute magnitudes */
      Serial.println("completed");
  
      Serial.println("final Data: ");
      for(int i = 0; i<128; i++){
        fft[i] = float(data[i]);
        Serial.println(fft[i]);
      }
      Serial.print("FFT len : ");
      Serial.println(sizeof(fft)/sizeof(fft[0]));
      Serial.println("Done");
    }
    
    else if(input=='4'){
      Serial.println("Predicted Data : " + sound[nn.predictClass(fft)]);
    }
  }
}
