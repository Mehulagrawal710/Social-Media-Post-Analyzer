import React from 'react';
import './App.css';

class Search extends React.Component {

    constructor(props) {
        super(props)
        this.state = {
            searchText : "",
        }
    }

    render() {
        
        return (
            <div id="search-box">
                <form className="row">
                    <div className="col-md-10">
                        <input onChange={this.handleSearchBoxChange} className="form-control form-control-lg" value={this.state.searchText} placeholder="Type Url Here" />
                    </div>
                    <div className="col-md-2" id="search-button">
                        <button className="btn btn-primary btn-lg form-group">Search</button>
                    </div>
                </form>
            </div>
        )

    }

    handleSearchBoxChange = (event) => {
        this.setState({
            searchText : event.target.value,
        })
    }

}

export default Search;