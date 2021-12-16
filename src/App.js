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

  useEffect(() => {
    SpeechRecognition.startListening({continuous: true})
  }, [])

  useEffect(() => {
    if (transcript.includes("reset")) {
      resetTranscript()
    } else {
      setText(transcript)
    }
  }, [transcript, finalTranscript, resetTranscript])

  useEffect(() => {
    axios.post('/convert_text_to_cdql', {"text": finalTranscript}).then(res => {
      setcdql(res["data"]["cdql"])
    });
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
            <Form.Label style={{ height: '100px', "font-size":"30px"}}>Transcript</Form.Label>
          </Form>
        </Col>
        <Col>
          <Form>
            <Form.Label>{text}</Form.Label>
          </Form>
        </Col>
        <Col xs="2"></Col>
      </Row>
      <Row>
        <Col xs="2"></Col>
        <Col xs="2">
            <Form>
              <Form.Label style={{ height: '100px', "font-size":"30px"}}>CDQL query</Form.Label>
            </Form>
          </Col>
          <Col>
            <Form>
              <Form.Label>{cdql}</Form.Label>
            </Form>
          </Col>
          <Col xs="2"></Col>
      </Row>
      <Row>
        <Col xs="2"></Col>
        <Col xs="2">
          <Form>
            <Form.Label style={{ height: '100px', "font-size":"30px"}}>Results</Form.Label>
          </Form>
        </Col>
        <Col>
          <Form>
            <Form.Label>{results}</Form.Label>
          </Form>
        </Col>
      </Row>
      <Col xs="2"></Col>
      
    </div>
  );
}

export default App;