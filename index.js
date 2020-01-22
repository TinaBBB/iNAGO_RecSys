// See https://github.com/dialogflow/dialogflow-fulfillment-nodejs
// for Dialogflow fulfillment library docs, samples, and to report issues
'use strict';
 
const functions = require('firebase-functions');
const {WebhookClient} = require('dialogflow-fulfillment');
const {Card, Suggestion} = require('dialogflow-fulfillment');
const{BasicCard, Button, Image} = require('actions-on-google');


const requestAPI = require('request-promise');
process.env.DEBUG = 'dialogflow:debug'; // enables lib debugging statements
 
exports.dialogflowFirebaseFulfillment = functions.https.onRequest((request, response) => {
  const agent = new WebhookClient({ request, response });
  const conv = agent.conv();
  console.log('Dialogflow Request headers: ' + JSON.stringify(request.headers));
  console.log('Dialogflow Request body: ' + JSON.stringify(request.body));
 
  function welcome(agent) {
    agent.add(`Welcome to my agent!`);
  }
 
  function fallback(agent) {
    agent.add(`I didn't understand`);
    agent.add(`I'm sorry, can you try again?`);
  }

  async function Sys_Recommend(agent){
    let response = await get_recommend(0);
    agent.add(response);
  }
  async function Sys_Critique_Star(agent){
    let response = await get_recommend(1);
    agent.add(response);
    }
  async function Sys_Critique_Price(agent){
    let response = await get_recommend(2);
    agent.add(response);
    }
  function saveData(data){
    if(conv !== null){
      console.log('conv is not null, save data')
      conv.data.RecData = data;
      }
    }
  function get_recommend(i){
    //https://ViolaS.api.stdlib.com/InitialRecommendation@dev/
    // agent.add('providing recommendations...');
    return requestAPI('https://violas.api.stdlib.com/initialrecommendation@dev/')
        .then(function(data){
          let initial_recommendation = JSON.parse(data);
          console.log(initial_recommendation);
          saveData(data);
          //conv.ask("?????????");
          let responseToUser = "";
          responseToUser += 'I recommend '+initial_recommendation.information[i].name + '. ';
          responseToUser += 'The average rating is ' + initial_recommendation.information[i].business_stars + '. ';
          responseToUser += 'The cuisine type is '+ initial_recommendation.information[i].categories + '. ';
          responseToUser += 'The price range is ' + initial_recommendation.information[i].price+'. ';
          responseToUser += 'Other people said '+ initial_recommendation.information[i].explanation+'. ';
          conv.ask(responseToUser);
          console.log(responseToUser);

          console.log(conv.data.RecData);
          if(conv.surface.capabilities.has('actions.capability.SCREEN_OUTPUT')){
              let image = 'https://raw.githubusercontent.com/jbergant/udemydemoimg/master/meetup.png';
                //if there's a screen avilable, add a card
              conv.ask(new BasicCard({
                  text: initial_recommendation.information[i].explanation,
                  subtitle: 'by ' + initial_recommendation.information[i].categories,
                  title: initial_recommendation.information[i].name,
                  image: new Image({
                      url: image,
                      alt: initial_recommendation.information[i].name,
                  }),
                  display: 'CROPPED',
                }));
            }
          return conv;

        }).catch(function (err) {
          console.log('No recommend data');
          console.log(err);
        });
  }

  // // Uncomment and edit to make your own intent handler
  // // uncomment `intentMap.set('your intent name here', yourFunctionHandler);`
  // // below to get this function to be run when a Dialogflow intent is matched
  // // below to get this function to be run when a Dialogflow intent is matched
  // function yourFunctionHandler(agent) {
  //   agent.add(`This message is from Dialogflow's Cloud Functions for Firebase editor!`);
  //   agent.add(new Card({
  //       title: `Title: this is a card title`,
  //       imageUrl: 'https://developers.google.com/actions/images/badges/XPM_BADGING_GoogleAssistant_VER.png',
  //       text: `This is the body text of a card.  You can even use line\n  breaks and emoji! 💁`,
  //       buttonText: 'This is a button',
  //       buttonUrl: 'https://assistant.google.com/'
  //     })
  //   );
  //   agent.add(new Suggestion(`Quick Reply`));
  //   agent.add(new Suggestion(`Suggestion`));
  //   agent.setContext({ name: 'weather', lifespan: 2, parameters: { city: 'Rome' }});
  // }

  // // Uncomment and edit to make your own Google Assistant intent handler
  // // uncomment `intentMap.set('your intent name here', googleAssistantHandler);`
  // // below to get this function to be run when a Dialogflow intent is matched
  // function googleAssistantHandler(agent) {
  //   let conv = agent.conv(); // Get Actions on Google library conv instance
  //   conv.ask('Hello from the Actions on Google client library!') // Use Actions on Google library
  //   agent.add(conv); // Add Actions on Google library responses to your agent's response
  // }
  // // See https://github.com/dialogflow/fulfillment-actions-library-nodejs
  // // for a complete Dialogflow fulfillment library Actions on Google client library v2 integration sample

  // Run the proper function handler based on the matched Dialogflow intent name
  let intentMap = new Map();
  intentMap.set('Default Welcome Intent', welcome);
  intentMap.set('Default Fallback Intent', fallback);
  intentMap.set('Request_for_Recommendation', Sys_Recommend);
  intentMap.set('User_Critique_Rating', Sys_Critique_Star);
  intentMap.set('User_Critique_Price', Sys_Critique_Price);

  // intentMap.set('your intent name here', yourFunctionHandler);
  // intentMap.set('your intent name here', googleAssistantHandler);
  agent.handleRequest(intentMap);
});
