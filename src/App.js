import './App.css';
import axios from 'axios';
import { useEffect, useState } from 'react';
import { Container, Row, Form, Col } from "react-bootstrap"
import 'bootstrap/dist/css/bootstrap.min.css';
import SpeechRecognition, { useSpeechRecognition } from 'react-speech-recognition'

function App() {
  const {
    transcript,
    finalTranscript,
    resetTranscript,
  } = useSpeechRecognition()
  const [text, setText] = useState("")
  const [cdql, setcdql] = useState("")
  const [results, setResults] = useState("")
  const wake_up_word = "Peter"

  function checkword(word) {
    return  word == wake_up_word;
  }
  const reset = () => {
    resetTranscript()
    setText()
    setcdql()
    setResults()
  }
  useEffect(() => {
    SpeechRecognition.startListening({continuous: true})
  }, [])

  useEffect(() => {
    if (transcript.includes("reset")) {
      reset()
    } else {
      if (transcript.includes(wake_up_word)) {
        const index = transcript.split(" ").findIndex(checkword)
        var output_transcript = transcript.split(" ").slice(index+1)
        output_transcript = output_transcript.join(" ")
        setText(output_transcript)
      }

    }
  }, [transcript, finalTranscript, resetTranscript])
  
  useEffect(() => {
    if (finalTranscript!= "" && finalTranscript.includes(wake_up_word)) {
      const index = finalTranscript.split(" ").findIndex(checkword)
      var output_finaltranscript = finalTranscript.split(" ").slice(index+1)
      output_finaltranscript = output_finaltranscript.join(" ")
      console.log(output_finaltranscript)

      axios.post('/convert_text_to_cdql', {"text": output_finaltranscript}).then(res => {
        setcdql(res["data"]["cdql"])
        resetInt = setTimeout(()=> {
          reset()
        }, 3000)
      })
    };

  }, [finalTranscript])

  useEffect(() => {
    
    axios.post('/convert_cdql_to_results', {"cdql": cdql}).then(res => {
      setResults(res["data"]["results"])
    });
  }, [cdql])

  return (
    <div style = {{margin:"0px"}}>
      <div>
        <img  src={'banner.png'} style={{width:"100%"}}/>
      </div>
      <div style={{ height: '100px' }}>
      </div>
      
      <Row>
        <Col xs="2"></Col>
        <Col xs="2">
          <Form>
            <Form.Label style={{ height: '200px', "font-size":"40px"}}>Transcript</Form.Label>
          </Form>
        </Col>
        <Col>
          <Form>
            <Form.Label style={{"font-size":"25px"}}>{text}</Form.Label>
          </Form>
        </Col>
        <Col xs="2"></Col>
      </Row>
      <Row>
        <Col xs="2"></Col>
        <Col xs="2">
            <Form>
              <Form.Label style={{ height: '200px', "font-size":"40px"}}>CDQL query</Form.Label>
            </Form>
          </Col>
          <Col>
            <Form>
              <Form.Label style={{"font-size":"25px"}} >{cdql}</Form.Label>
            </Form>
          </Col>
          <Col xs="2"></Col>
      </Row>
      <Row>
        <Col xs="2"></Col>
        <Col xs="2">
          <Form>
            <Form.Label style={{ height: '200px', "font-size":"40px"}}>Results</Form.Label>
          </Form>
        </Col>
        <Col>
          <Form>
            <Form.Label style={{"font-size":"25px"}}>{results}</Form.Label>
          </Form>
        </Col>
      </Row>
      <Col xs="2"></Col>
      
    </div>
  );
}

export default App;