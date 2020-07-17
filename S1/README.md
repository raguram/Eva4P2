## Problem Statement 

This assignment is aimed at setting up the AWS lambda for ML deep learning prediction with MobileNetV2 model. 

## Solution 

As directed by the steps, I used Serverless to manage the packaging and deployment of the lambda function. To overcome 256MB limitation of lambda package, 
I added the wheel file (.whl) for pytorch along with minimum dependencies to the requirements.txt file. 

## Endpoint details 

The lambda endpoint is hosted at: https://2fmno3fn46.execute-api.ap-south-1.amazonaws.com/dev/classify

Use the below request to invoke the endpoint 

<pre>
curl --location --request POST 'https://2fmno3fn46.execute-api.ap-south-1.amazonaws.com/dev/classify
' \
--header 'content-type: multipart/form-data' \
--form '=image_file_name.jpeg'
</pre>

## Screen shot

![alt screen-shot](https://github.com/raguram/Eva4P2/blob/master/S1/MobileNetV2Service/output-screenshot.png)


