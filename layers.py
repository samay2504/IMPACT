import tensorflow as tf
import json

def read_tflite_model(tflite_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    # Extract the model details
    model_details = {
         'inputs': [],
         'outputs': [],
         'layers': []
     }

    input_details = interpreter.get_input_details()
    for detail in input_details:
        model_details['inputs'].append({
             'name': detail['name'],
             'shape': detail['shape'].tolist(),
             'dtype': str(detail['dtype']).split("'")[1]
         })

    output_details = interpreter.get_output_details()
    for detail in output_details:
        model_details['outputs'].append({
             'name': detail['name'],
             'shape': detail['shape'].tolist(),
             'dtype': str(detail['dtype']).split("'")[1]
         })

    # Read the model's layers from the TensorFlow Lite model
    def get_tensor_details(tensor_index):
        tensor = interpreter.get_tensor_details()[tensor_index]
        return {
             'shape': tensor['shape'].tolist(),
             'dtype': str(tensor['dtype']).split("'")[1]
         }

    for i in range(len(interpreter.get_tensor_details())):
        tensor = interpreter.get_tensor_details()[i]
        layer_info = {
             'name': tensor['name'],
             'index': tensor['index'],
             'shape': get_tensor_details(tensor['index'])['shape'],
             'dtype': get_tensor_details(tensor['index'])['dtype']
         }
        model_details['layers'].append(layer_info)

    return model_details

# Example usage
tflite_path = 'D:/Projects/Impact hp/2.tflite'
model_details = read_tflite_model(tflite_path)

# Save the model details to a JSON file for inspection
with open('model_details.json', 'w') as f:
    json.dump(model_details, f, indent=4)