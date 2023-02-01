"""
Filter inference results based on confidence
"""
import json

THRESHOLD = 0.500
def lambda_handler(event, context):
    
    # Grab the inferences from the event
   
    inferences = event["body"]['inferences']
    
    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = max(list(inferences))>THRESHOLD     ## TODO: fill in (True, if a value exists above 'THRESHOLD')
    
    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        pass
    else:
        raise Exception("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        'statusCode': 200,
        'body': event            ## Passing the final event as a python dictionary
    }
    
