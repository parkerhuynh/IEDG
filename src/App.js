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
    <Container>
      <div>
        <img  src={'banner.png'} />
      </div>
      <div style={{ height: '100px' }}>
      </div>
      <Row style = {{ border: "1px solid black"}}>
        <Col xs="2">
          <Form>
            <Form.Label style={{ height: '100px' }}>Transcript</Form.Label>
          </Form>
        </Col>
        <Col>
          <Form>
            <Form.Label>{text}</Form.Label>
          </Form>
        </Col>
      </Row>
      <Row style = {{ border: "1px solid black"}}>
      <Col xs="2">
          <Form>
            <Form.Label style={{ height: '100px' }}>CDQL query</Form.Label>
          </Form>
        </Col>
        <Col>
          <Form>
            <Form.Label>{cdql}</Form.Label>
          </Form>
        </Col>
      </Row>
      <Row style = {{ border: "1px solid black"}}>
      <Col xs="2">
          <Form>
            <Form.Label style={{ height: '100px' }}>Results</Form.Label>
          </Form>
        </Col>
        <Col>
          <Form>
            <Form.Label>{results}</Form.Label>
          </Form>
        </Col>
      </Row>
      
    </Container>
  );
}

export default App;