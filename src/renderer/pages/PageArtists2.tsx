import { Link, useNavigate } from 'react-router-dom';
import axios from 'axios';
import { useState } from 'react';
import Recommending from './recommending';

const PageArtists2 = () => {
  const navigate = useNavigate();

  const [data, setData] = useState({});
  const [name, setName] = useState('');
  const [no, setNo] = useState('');
  const [artList, setArtList] = useState();
  const [clicked, setClicked] = useState(false);

  const ShowRecButton = (listedBool) => {
    if (listedBool.listedData == true) {
      return (
        <Link to="/RecArtists" state={{ artist: name, list: artList }}>
          <button>Reveal Recommended Artists</button>
        </Link>
      );
    } else if (listedBool.listedData == false) {
      setClicked(false);
      return (
        <h2>
          Oh no! There seems to be an error, please check your inputs or try
          another artist
        </h2>
      );
    } else if (listedBool.listedData == undefined && clicked) {
      return <Recommending />;
    }
  };

  const submitArtist = async (data: any) => {
    try {
      const response = await axios.post('http://127.0.0.1:5000/Artists2', data);
      setData(response.data);
      setArtList(response.data.list);
    } catch (error) {
      console.error(error);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    console.log(name, no);
    setData({});
    submitArtist({ name: name, number: no });
    setClicked(true);
  };

  return (
    <div>
      <div>
        <button className="back" onClick={() => navigate(-1)}>
          Back
        </button>
      </div>

      <div className="titleHeader menuMain">
        <h1 className="Title">SpecoMe</h1>
        <h3>A Spotify Recommendation App</h3>
      </div>

      <form onSubmit={handleChange}>
        <label for="artist">
          <h2>Who is the artist?</h2>
        </label>
        <input
          type="text"
          id="artist"
          placeholder="My favourite artist..."
          onChange={(e) => setName(e.target.value)}
        ></input>

        <label for="number">
          <h2>Number of artists to generate?</h2>
        </label>
        <input
          type="text"
          id="number"
          placeholder="e.g. 10"
          onChange={(e) => setNo(e.target.value)}
        ></input>

        <div className="formButton">
          {!clicked && <button type="submit">Recommend!</button>}
        </div>

        <div>
          <ShowRecButton listedData={data.listed} />
        </div>
      </form>
    </div>
  );
};
export default PageArtists2;
