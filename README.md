# Industry Engagement and Demonstrators Grant: IEDG
Conversational Natural Language Interface for Smart Homes

 

Smart homes are increasing in complexity and ability. Interacting with a smart home through natural language is increasingly important. The load is increasingly on the user to remember an increasingly large set of home assistant commands. This interaction can be improved by conversational systems that have a longer series of interactions with the user. The project will build a demonstration of a smart home which can be controlled by a natural language conversation system.

The demonstration will show a user interacting with a smart home with a series of very complex conversations. Such as asking “What is the temperature in the living room”.  The system will need to understand this question, discover IoT services, then compose context queries to be to understand the context. The system may then offer options for improving the room temperature.

To achieve this, there are a series of steps. These include, speech to text, text to CDQL. To implement this the student will need to create a GUI, use IoT technologies, NLP technologies, use ontologies, adapt existing middleware such as Dialogflow, and integrate with existing systems including CoaaS and Home Assistant. 

The following steps aim to deploy a user interface for smarthome.
## Install

Clone the repository.

`pip install -r requirements.txt`

**npm** is required to integrate the user interface. Depending on the operating system such as Windown, Linux, and Mac, the method to install **npm** can be found at the following link:

https://docs.npmjs.com/downloading-and-installing-node-js-and-npm

## Build a text to CDQL

The purpose of this section is to crate a text-to-CDQL system.
 
### Data Generator

Data Generator aims to create the data to train and evaluate the model. You can directly download the pre-generated data from [here](https://github.com/parkerhuynh/IEDG/tree/main/text-to-CDQL/data). If you want to create by yourself, you can edit the templates into the file named `data_templates.py` in the folder named `text-to-CDQL`. After that, you run the following code to generate the dataset:

`python text-to-CDQL\data_generator.py`

### Training Model

Before trainning the model, you should set the parameters of your model such as the number of hidden units, epoches, and learning rate in the file `config.py` in the folder named `text-to-CDQL`. To train the model, run:

`python text-to-CDQL\training.py`

You can also load the pre-trained model from [here](https://github.com/parkerhuynh/IEDG/tree/main/text-to-CDQL/saved_model/translator).

### `npm start`

Runs the app in the development mode.\
Open [http://localhost:3000](http://localhost:3000) to view it in the browser.

The page will reload if you make edits.\
You will also see any lint errors in the console.

### `npm test`

Launches the test runner in the interactive watch mode.\
See the section about [running tests](https://facebook.github.io/create-react-app/docs/running-tests) for more information.

### `npm run build`

Builds the app for production to the `build` folder.\
It correctly bundles React in production mode and optimizes the build for the best performance.

The build is minified and the filenames include the hashes.\
Your app is ready to be deployed!

See the section about [deployment](https://facebook.github.io/create-react-app/docs/deployment) for more information.

### `npm run eject`

**Note: this is a one-way operation. Once you `eject`, you can’t go back!**

If you aren’t satisfied with the build tool and configuration choices, you can `eject` at any time. This command will remove the single build dependency from your project.

Instead, it will copy all the configuration files and the transitive dependencies (webpack, Babel, ESLint, etc) right into your project so you have full control over them. All of the commands except `eject` will still work, but they will point to the copied scripts so you can tweak them. At this point you’re on your own.

You don’t have to ever use `eject`. The curated feature set is suitable for small and middle deployments, and you shouldn’t feel obligated to use this feature. However we understand that this tool wouldn’t be useful if you couldn’t customize it when you are ready for it.

## Learn More

You can learn more in the [Create React App documentation](https://facebook.github.io/create-react-app/docs/getting-started).

To learn React, check out the [React documentation](https://reactjs.org/).

### Code Splitting

This section has moved here: [https://facebook.github.io/create-react-app/docs/code-splitting](https://facebook.github.io/create-react-app/docs/code-splitting)

### Analyzing the Bundle Size

This section has moved here: [https://facebook.github.io/create-react-app/docs/analyzing-the-bundle-size](https://facebook.github.io/create-react-app/docs/analyzing-the-bundle-size)

### Making a Progressive Web App

This section has moved here: [https://facebook.github.io/create-react-app/docs/making-a-progressive-web-app](https://facebook.github.io/create-react-app/docs/making-a-progressive-web-app)

### Advanced Configuration

This section has moved here: [https://facebook.github.io/create-react-app/docs/advanced-configuration](https://facebook.github.io/create-react-app/docs/advanced-configuration)

### Deployment

This section has moved here: [https://facebook.github.io/create-react-app/docs/deployment](https://facebook.github.io/create-react-app/docs/deployment)

### `npm run build` fails to minify

This section has moved here: [https://facebook.github.io/create-react-app/docs/troubleshooting#npm-run-build-fails-to-minify](https://facebook.github.io/create-react-app/docs/troubleshooting#npm-run-build-fails-to-minify)
